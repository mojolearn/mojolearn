# SPDX-License-Identifier: Apache-2.0
"""Immutable native entry tables published by loaded extension images.

Publication and lookup are small metadata operations, with no independent
numerical work to split. Providers publish only after their module is ready;
callers retain image pins at UNPACK/PACK. RUN never owns a Python reference.

THE CALL PATH (lane entry-table-zero-alloc, 2026-11-11). A table is read ONCE,
when a caller's connection owner is built (import-time admission or estimator
construction): `entry_table_resolve` does the named-slot lookup, the acquire
load, the key and schema checks and every refusal, and returns an `EntryRef`
(the thin C pointer plus a layout word, two machine words). Every later call
is `entry_ref_call`: one indirect call over caller-pinned spans, with the
output words on the caller's stack. No `String`, no `List`, no slot lookup,
no `Optional`, no re-check per call; a refusal text is built only when the
callee reports one. `entry_table_read` keeps its name and its error texts and
now returns `entry_table_resolve(...).call`.
"""
from std.atomic import Atomic, Ordering
from std.sys import is_defined
from std.sys.info import size_of
from core.native_call_output import NativeCallOutput
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call_spans, native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime ENTRY_TABLE_LAYOUT = 1


struct NativeEntryTable(Defaultable, Movable):
    var synchronization: List[Int64]
    var mode: Int
    var vendor: Int
    var publisher: Int
    var schemas: List[String]
    var operations: List[Int]
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.mode = -1
        self.vendor = -1
        self.publisher = 0
        self.schemas = List[String]()
        self.operations = List[Int]()
        self.entries = List[NativeDescriptorCall]()


def entry_table_publish[name: StaticString](mode: Int, vendor: Int,
    publisher: Int, schemas: List[String], operations: List[Int],
    entries: List[NativeDescriptorCall]) raises:
    if (mode < 0 or mode > 1 or vendor < 0 or vendor > 3 or not publisher
            or not len(entries) or len(entries) != len(schemas)
            or len(entries) != len(operations)):
        raise Error("native entry table: invalid publication")
    for index in range(len(entries)):  # driver-loop: fixed ABI entries
        if schemas[index].byte_length() == 0 or operations[index] <= 0:
            raise Error("native entry table: invalid entry schema")
    comptime SLOT = SharedSlot[NativeEntryTable, name, ENTRY_TABLE_LAYOUT]
    var slot = SLOT.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        if Atomic.load[ordering=Ordering.ACQUIRE](lock + 1) != 0:
            if (slot[].mode != mode or slot[].vendor != vendor
                    or slot[].publisher != publisher or len(slot[].entries) != len(entries)):
                raise Error("native entry table: conflicting publication: " + String(name))
            for index in range(len(entries)):  # driver-loop: fixed ABI entries
                if slot[].schemas[index] != schemas[index] or slot[].operations[index] != operations[index]:
                    raise Error("native entry table: incompatible publication: " + String(name))
            # A repeated module initialization must publish the same thin C
            # pointers, not silently replace a live image under existing calls.
            var prior = slot[].entries.unsafe_ptr().bitcast[UInt8]()
            var incoming = entries.unsafe_ptr().bitcast[UInt8]()
            for byte in range(len(entries) * size_of[NativeDescriptorCall]()):
                if prior[byte] != incoming[byte]:
                    raise Error("native entry table: endpoint differs: " + String(name))
            return
        slot[].mode = mode
        slot[].vendor = vendor
        slot[].publisher = publisher
        slot[].schemas = schemas.copy()
        slot[].operations = operations.copy()
        slot[].entries = entries.copy()
        # Everything read after the acquire below is immutable for the image
        # lifetime. No reader observes a partially filled table.
        Atomic.store[ordering=Ordering.RELEASE](lock + 1, Int64(1))
    finally:
        shared_slot_unlock(lock)


@always_inline
def entry_ref_word(mode: Int, vendor: Int, operations: Int) -> Int:
    """The layout word of a resolved entry: the table layout, the operation
    count the caller admitted, its tier and its vendor. Two refs with the same
    pointer and word came from the same publication."""
    return ((ENTRY_TABLE_LAYOUT & 0xFFFF) << 48) | ((operations & 0xFFFFFFFF) << 16) | ((mode & 0xFF) << 8) | (vendor & 0xFF)


@always_inline
def entry_ref_vendor(entry: EntryRef) -> Int:
    """The vendor word the ref was resolved for (0 CPU, 1 Metal, 2 CUDA, 3 HIP)."""
    return entry.word & 0xFF


@always_inline
def entry_ref_mode(entry: EntryRef) -> Int:
    """The tier word the ref was resolved for (1 identical, 0 fast)."""
    return (entry.word >> 8) & 0xFF


@fieldwise_init
struct EntryRef(ImplicitlyCopyable, Movable):
    """A table entry resolved once: the thin C pointer and its layout word.
    A connection owner holds it; a call through it is one indirect call."""

    var call: NativeDescriptorCall
    var word: Int


def _entry_unresolved(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    native_descriptor_report(Error("native entry table: entry is unavailable: not resolved for this owner"),
                             context, sink)
    return 1


def entry_ref_unresolved() -> EntryRef:
    """The ref an owner holds for a table its kind never calls (word 0): a
    call through it refuses; it never reads a slot."""
    return EntryRef(_entry_unresolved, 0)


def entry_table_resolve[name: StaticString](mode: Int, vendor: Int,
    index: Int, schema: String, operations: Int) raises -> EntryRef:
    """Read the published slot once and admit it. Every refusal of the table
    (not published, key differs, entry unavailable, ABI schema differs)
    happens here, never at call time."""
    comptime SLOT = SharedSlot[NativeEntryTable, name, ENTRY_TABLE_LAYOUT]
    var slot = SLOT.get_or_create_ptr()
    var published = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].synchronization.unsafe_ptr() + 1)
    if Atomic.load[ordering=Ordering.ACQUIRE](published) != 1:
        raise Error("native entry table: provider is not published: " + String(name))
    if slot[].mode != mode or slot[].vendor != vendor:
        raise Error("native entry table: execution key differs: " + String(name))
    if index < 0 or index >= len(slot[].entries):
        raise Error("native entry table: entry is unavailable: " + String(name))
    if slot[].schemas[index] != schema or slot[].operations[index] != operations:
        raise Error("native entry table: ABI schema differs: " + String(name))
    return EntryRef(slot[].entries[index], entry_ref_word(mode, vendor, operations))


def entry_table_read[name: StaticString](mode: Int, vendor: Int,
    index: Int, schema: String, operations: Int) raises -> NativeDescriptorCall:
    """The resolved pointer alone (resolution-time callers and tests)."""
    return entry_table_resolve[name](mode, vendor, index, schema, operations).call


@always_inline
def entry_ref_call(entry: EntryRef, operation: Int, addresses: Int, address_count: Int,
    ints: Int, int_count: Int, floats: Int, float_count: Int) raises -> Int:
    """One call through a resolved entry over caller-pinned spans (address,
    count). No heap allocation: the output words live on this frame and are
    passed by reference to the out-of-line span call, so their address is
    real storage for the whole foreign call. The error list stays empty (no
    allocation) unless the callee reports a refusal."""
    comptime if is_defined["MOJOLEARN_ENTRY_TABLE_HEAP_OUTPUT"]():
        # Opt-in former form (see core/native_descriptor_call.mojo).
        var held = List[NativeCallOutput]()
        held.append(NativeCallOutput())
        var held_result = native_descriptor_call_spans(entry.call, operation, addresses, address_count,
            ints, int_count, floats, float_count, held[0])
        _ = held
        return held_result
    else:
        var output = NativeCallOutput()
        var result = native_descriptor_call_spans(entry.call, operation, addresses, address_count,
            ints, int_count, floats, float_count, output)
        _ = output
        return result


def entry_ref_call(entry: EntryRef, operation: Int, addresses: List[Int], ints: List[Int],
    floats: List[Float64]) raises -> Int:
    """`entry_ref_call` over request Lists the caller already owns."""
    var result = entry_ref_call(entry, operation, Int(addresses.unsafe_ptr()), len(addresses),
        Int(ints.unsafe_ptr()), len(ints), Int(floats.unsafe_ptr()), len(floats))
    # Raw span addresses do not themselves retain their List owners.
    _ = addresses
    _ = ints
    _ = floats
    return result


def entry_ref_call[NA: Int, NP: Int, NF: Int](entry: EntryRef, operation: Int,
    addresses: Array[Int, NA], ints: Array[Int, NP], floats: Array[Float64, NF]) raises -> Int:
    """`entry_ref_call` over fixed request envelopes held inline on the
    caller's frame (`var a: Array[Int, 2] = [x, y]`): no heap at all."""
    var aa = 0
    var pp = 0
    var ff = 0
    comptime if NA > 0:
        aa = Int(addresses.unsafe_ptr())
    comptime if NP > 0:
        pp = Int(ints.unsafe_ptr())
    comptime if NF > 0:
        ff = Int(floats.unsafe_ptr())
    var result = entry_ref_call(entry, operation, aa, NA, pp, NP, ff, NF)
    # Integer addresses carry no lifetime: keep the envelopes past the call.
    _ = addresses
    _ = ints
    _ = floats
    return result


def entry_ref_same(left: EntryRef, right: EntryRef) -> Bool:
    """True when both refs hold the same pointer and layout word (tests: a
    connection owner reuses one resolved ref across calls)."""
    return left.word == right.word and entry_table_same_call(left.call, right.call)


def entry_table_same_call(left: NativeDescriptorCall, right: NativeDescriptorCall) -> Bool:
    # Native thin C function pointers have no owned captures. This comparison
    # is import-time admission metadata, never a numerical/prediction loop.
    var lp = Pointer(to=left).bitcast[UInt8]()
    var rp = Pointer(to=right).bitcast[UInt8]()
    var same = True
    for byte in range(size_of[NativeDescriptorCall]()):
        if lp[byte] != rp[byte]:
            same = False
    _ = left
    _ = right
    return same
