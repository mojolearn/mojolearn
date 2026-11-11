# SPDX-License-Identifier: Apache-2.0
"""Immutable native entry tables published by loaded extension images.

Publication and lookup are small metadata operations, with no independent
numerical work to split. Providers publish only after their module is ready;
callers retain image pins at UNPACK/PACK. RUN never owns a Python reference.
"""
from std.atomic import Atomic, Ordering
from std.sys.info import size_of
from core.native_descriptor_abi import NativeDescriptorCall
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
        if not len(schemas[index]) or operations[index] <= 0:
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


def entry_table_read[name: StaticString](mode: Int, vendor: Int,
    index: Int, schema: String, operations: Int) raises -> NativeDescriptorCall:
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
    return slot[].entries[index]
