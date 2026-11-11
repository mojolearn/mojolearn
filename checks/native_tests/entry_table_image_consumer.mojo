# SPDX-License-Identifier: Apache-2.0
"""Test-only shared image B: calls A exclusively via table and native handle.

The provider body is intentionally absent from this source closure. The root
controller retains both loaded images and runs dependent lifecycle steps in
order. Independent controls use separate processes; no GPU context is created.
"""
from std.memory import ArcPointer
from core.entry_table import NativeEntryTable, entry_table_read
from core.native_descriptor_call import native_descriptor_call
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.state_registry import StateHandle
from checks.native_tests.entry_table_image_common import IMAGE_TABLE, IMAGE_ABI, IMAGE_STATES, ImageState


struct HeldImageState(Defaultable, Movable):
    var lock: List[Int64]
    var leases: List[ArcPointer[ImageState]]

    def __init__(out self):
        self.lock = List[Int64](length=1, fill=0)
        self.leases = List[ArcPointer[ImageState]]()


comptime HELD = SharedSlot[HeldImageState, "MojolearnTestCrossImageLeaseV1"]


def _held_lease() raises -> ArcPointer[ImageState]:
    var slot = HELD.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].lock.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        if not len(slot[].leases):
            raise Error("cross-image native lease is absent")
        return slot[].leases[0].copy()
    finally:
        shared_slot_unlock(lock)


def _query(lease: ArcPointer[ImageState], increment: Int) raises -> Int:
    var entry = entry_table_read[IMAGE_TABLE](1, 0, 0, String(IMAGE_ABI), 1)
    var result = native_descriptor_call(entry, 0, [], [lease[].values[0], increment], [])
    _ = lease
    return result


@export
def native_entry_image_query(index: Int, generation: UInt64, increment: Int) abi("C") -> Int:
    try:
        var lease = IMAGE_STATES.acquire(StateHandle(index, generation))
        return _query(lease, increment)
    except problem:
        print("REFUSE cross-image query:", String(problem))
        return -700


@export
def native_entry_image_hold(index: Int, generation: UInt64) abi("C") -> Int:
    try:
        var lease = IMAGE_STATES.acquire(StateHandle(index, generation))
        var retired = Optional[ArcPointer[ImageState]]()
        var slot = HELD.get_or_create_ptr()
        var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].lock.unsafe_ptr())
        shared_slot_lock(lock)
        try:
            if len(slot[].leases):
                retired = slot[].leases.pop()
            slot[].leases.append(lease.copy())
        finally:
            shared_slot_unlock(lock)
            _ = retired^
            _ = lease
        return 0
    except problem:
        print("FAIL cross-image hold:", String(problem))
        return 1


@export
def native_entry_image_held(increment: Int) abi("C") -> Int:
    try:
        var lease = _held_lease()
        return _query(lease, increment)
    except problem:
        print("REFUSE cross-image held query:", String(problem))
        return -700


@export
def native_entry_image_drop() abi("C") -> Int:
    try:
        var retired = Optional[ArcPointer[ImageState]]()
        var slot = HELD.get_or_create_ptr()
        var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].lock.unsafe_ptr())
        shared_slot_lock(lock)
        try:
            if len(slot[].leases):
                retired = slot[].leases.pop()
        finally:
            shared_slot_unlock(lock)
            _ = retired^
        return 0
    except:
        return 1


@export
def native_entry_image_schema_refusal() abi("C") -> Int:
    try:
        _ = entry_table_read[IMAGE_TABLE](1, 0, 0, String("mojolearn.test.cross-image.v2"), 1)
        return 0
    except:
        return 1


@export
def native_entry_image_layout_abort() abi("C") -> Int:
    # Intentional incompatible layout read in a dedicated subprocess only.
    # The SharedSlot diagnostic must abort before any mismatched value is read.
    try:
        comptime WRONG = SharedSlot[NativeEntryTable, IMAGE_TABLE, 2]
        _ = WRONG.get_or_create_ptr()
        return 0
    except:
        return 1
