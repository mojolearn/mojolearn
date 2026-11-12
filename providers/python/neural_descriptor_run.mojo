# SPDX-License-Identifier: Apache-2.0
"""Typed neural descriptor provider records (RUN, no Python).

A connected provider is two parts. The door
(`providers/python/neural_descriptor_pyapi_door.mojo`) owns the Python pins: the
compiled module, its capsule and its factory entry, held by the boxed
`NeuralDescriptorProvider`. This module owns the typed record those pins keep
valid: the descriptor table entry, the ABI name, the per-operation call counts
and the retained call workspace. The record sits in a one-element heap list
inside the pinned provider, so its address is stable for the provider's
lifetime; that address is the provider id word a RUN caller holds. While the
door keeps the provider pinned, a RUN caller executes by id with no Python
object, downcast or lookup.

Counting and admission (`neural_descriptor_record_admit`) run where the caller
holds the GIL; the call itself (`neural_descriptor_record_call`) may run with
the GIL released.
"""
from std.memory import unsafe_memcpy
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call_spans, native_descriptor_leased_call
from core.native_call_output import NativeCallWorkspace, NativeCallWorkspaceLease
from core.shared_slot import SharedSlot


struct NeuralDescriptorRecord(Movable):
    var name: String
    var entry: NativeDescriptorCall
    var calls: List[Int]
    # Heap-backed ABI output and session-prefix capacity retained with this
    # admitted provider. Its exclusive lease handles concurrent/reentrant
    # calls without sharing writable scratch.
    var workspace: List[NativeCallWorkspace]

    def __init__(out self, var name: String, entry: NativeDescriptorCall, operation_count: Int):
        self.name = name^
        self.entry = entry
        self.calls = List[Int](length=operation_count, fill=0)
        self.workspace = List[NativeCallWorkspace]()
        self.workspace.append(NativeCallWorkspace())


comptime NeuralDescriptorRecordPtr = MutPointer[NeuralDescriptorRecord, MutUntrackedOrigin]


def neural_descriptor_record(provider_id: Int) -> NeuralDescriptorRecordPtr:
    # The caller's door keeps the provider that owns this record pinned.
    return NeuralDescriptorRecordPtr(unsafe_from_address=provider_id)


def neural_descriptor_record_entry(provider_id: Int) -> NativeDescriptorCall:
    return neural_descriptor_record(provider_id)[].entry


def neural_descriptor_record_count(provider_id: Int) -> Int:
    return len(neural_descriptor_record(provider_id)[].calls)


def neural_descriptor_record_workspace(provider_id: Int) -> Int:
    return Int(neural_descriptor_record(provider_id)[].workspace.unsafe_ptr())


def neural_descriptor_record_admit(provider_id: Int, operation: Int) raises:
    """Refuse an operation outside the admitted table and count the attempt."""
    var record = neural_descriptor_record(provider_id)
    if operation < 0 or operation >= len(record[].calls):
        raise Error("neural native ABI: unknown operation")
    record[].calls[operation] += 1


def neural_descriptor_record_call(
    provider_id: Int, operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64],
) raises -> Int:
    """One synchronous call of an admitted operation under the record's
    workspace lease. Safe with the GIL released; counts nothing."""
    var record = neural_descriptor_record(provider_id)
    return native_descriptor_leased_call(record[].entry, Int(record[].workspace.unsafe_ptr()),
        operation, addresses, ints, floats)


def neural_descriptor_record_execute(
    provider_id: Int, operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64],
) raises -> Int:
    """Admit, count and call: the typed route for RUN callers holding an id."""
    neural_descriptor_record_admit(provider_id, operation)
    return neural_descriptor_record_call(provider_id, operation, addresses, ints, floats)


def neural_descriptor_reserved_begin(
    provider_id: Int, handle: Int, reserve: Int, release: Int, operation: Int, ints: List[Int],
) raises -> NativeCallWorkspaceLease:
    """Lease the record's workspace, stage `handle` + `ints` as the launch
    words and run the session reserve operation (caller holds the GIL). The
    lease's destructor releases it if this raises."""
    var record = neural_descriptor_record(provider_id)
    var count = len(record[].calls)
    if reserve < 0 or reserve >= count or release < 0 or release >= count or operation < 0 or operation >= count:
        raise Error("neural native ABI: unknown operation")
    var loan = NativeCallWorkspaceLease(Int(record[].workspace.unsafe_ptr()))
    var scratch = loan.pointer()
    # One exclusive vector supplies both the session-only reserve/release
    # span and the full prefixed operation span. Refresh every launch word;
    # only allocation capacity survives calls, never argument values.
    scratch[].words.resize(len(ints) + 1, 0)
    scratch[].words[0] = handle
    if len(ints) > 0:
        unsafe_memcpy(dest=scratch[].words.unsafe_ptr() + 1, src=ints.unsafe_ptr(), count=len(ints))
    var words = Int(scratch[].words.unsafe_ptr())
    record[].calls[reserve] += 1
    _ = native_descriptor_call_spans(record[].entry, reserve, 0, 0, words, 1, 0, 0, scratch[].output)
    # Counted here, under the GIL, for the call the door makes next.
    record[].calls[operation] += 1
    _ = ints
    return loan^


def neural_descriptor_reserved_operation(
    provider_id: Int, mut loan: NativeCallWorkspaceLease, operation: Int,
    addresses: List[Int], floats: List[Float64],
) raises -> Int:
    """The reserved operation over the staged words (GIL may be released)."""
    var record = neural_descriptor_record(provider_id)
    var scratch = loan.pointer()
    var result = native_descriptor_call_spans(record[].entry, operation,
        Int(addresses.unsafe_ptr()), len(addresses), Int(scratch[].words.unsafe_ptr()), len(scratch[].words),
        Int(floats.unsafe_ptr()), len(floats), scratch[].output)
    _ = addresses
    _ = floats
    return result


def neural_descriptor_reserved_end(provider_id: Int, mut loan: NativeCallWorkspaceLease, release: Int) raises:
    """Run the session release operation (caller holds the GIL)."""
    var record = neural_descriptor_record(provider_id)
    var scratch = loan.pointer()
    record[].calls[release] += 1
    _ = native_descriptor_call_spans(record[].entry, release, 0, 0,
        Int(scratch[].words.unsafe_ptr()), 1, 0, 0, scratch[].output)


struct NeuralDescriptorAdmitted(Defaultable, Movable):
    # Binding modules whose whole namespace was admitted (every key an exact
    # string), as (module address, dictionary address, dictionary size). Plain
    # integers only: no Python reference outlives the interpreter here.
    var modules: List[Int]
    var dicts: List[Int]
    var sizes: List[Int]

    def __init__(out self):
        self.modules = List[Int]()
        self.dicts = List[Int]()
        self.sizes = List[Int]()


comptime NEURAL_DESCRIPTOR_ADMITTED = SharedSlot[NeuralDescriptorAdmitted, "MojolearnNeuralDescriptorAdmittedV2"]

#: `neural_descriptor_admitted_find` result for an unchanged admitted namespace.
comptime NEURAL_DESCRIPTOR_ADMITTED_HIT = -2


def neural_descriptor_admitted_find(module_address: Int, dict_address: Int, size: Int) raises -> Int:
    """NEURAL_DESCRIPTOR_ADMITTED_HIT when this module's namespace was admitted
    at the same dictionary and size; else the module's stale slot, or -1."""
    var record = NEURAL_DESCRIPTOR_ADMITTED.get_or_create_ptr()
    for i in range(len(record[].modules)):  # small-loop(record: admitted binding modules): a few compiled extension modules per process, never data
        if record[].modules[i] == module_address:
            if record[].dicts[i] == dict_address and record[].sizes[i] == size:
                return NEURAL_DESCRIPTOR_ADMITTED_HIT
            return i
    return -1


def neural_descriptor_admitted_store(slot: Int, module_address: Int, dict_address: Int, size: Int) raises:
    """Record a namespace that just passed the full key walk."""
    var record = NEURAL_DESCRIPTOR_ADMITTED.get_or_create_ptr()
    if slot >= 0:
        record[].dicts[slot] = dict_address
        record[].sizes[slot] = size
        return
    if len(record[].modules) >= 64:
        record[].modules.clear()
        record[].dicts.clear()
        record[].sizes.clear()
    record[].modules.append(module_address)
    record[].dicts.append(dict_address)
    record[].sizes.append(size)
