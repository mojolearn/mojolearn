# SPDX-License-Identifier: Apache-2.0
"""Typed process-wide state handles with generation-checked native leases.

Only registry metadata is locked. An acquired Arc retains native state across
removal; its destruction and all numerical work happen outside the lock.
Borrowed buffers inside a state still require the owner's documented pins.
"""
from std.collections import Dict
from std.memory import ArcPointer
from std.sys.info import size_of
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime STATE_REGISTRY_LAYOUT = 1


@fieldwise_init
struct StateHandle(Copyable, Movable):
    var index: Int
    var generation: UInt64


struct StateRegistryStorage[T: Movable & Deinitable](Defaultable, Movable):
    var state_size: Int
    var lock: List[Int64]
    var values: Dict[Int, ArcPointer[Self.T]]
    var generations: List[UInt64]
    var released: List[Int]

    def __init__(out self):
        self.state_size = size_of[Self.T]()
        self.lock = List[Int64](length=1, fill=0)
        self.values = Dict[Int, ArcPointer[Self.T]]()
        self.generations = List[UInt64]()
        self.released = List[Int]()


struct StateRegistry[T: Movable & Deinitable, name: StaticString]:
    comptime SLOT = SharedSlot[StateRegistryStorage[Self.T], Self.name, STATE_REGISTRY_LAYOUT]

    @staticmethod
    def insert(var state: Self.T) raises -> StateHandle:
        # Native state construction/allocation is outside the metadata lock.
        var lease = ArcPointer[Self.T](state^)
        var slot = Self.SLOT.get_or_create_ptr()
        if slot[].state_size != size_of[Self.T]():
            raise Error("native state registry: state layout differs")
        var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].lock.unsafe_ptr())
        shared_slot_lock(lock)
        try:
            var index = len(slot[].generations)
            if len(slot[].released):
                index = slot[].released.pop()
                # A generation is never allowed to wrap and revive a handle.
                if slot[].generations[index] == UInt64.MAX:
                    index = len(slot[].generations)
                else:
                    slot[].generations[index] += 1
            if index == len(slot[].generations):
                slot[].generations.append(UInt64(1))
            slot[].values[index] = lease.copy()
            return StateHandle(index, slot[].generations[index])
        finally:
            shared_slot_unlock(lock)
            _ = lease

    @staticmethod
    def acquire(handle: StateHandle) raises -> ArcPointer[Self.T]:
        var slot = Self.SLOT.get_or_create_ptr()
        if slot[].state_size != size_of[Self.T]():
            raise Error("native state registry: state layout differs")
        var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].lock.unsafe_ptr())
        shared_slot_lock(lock)
        try:
            if (handle.index < 0 or handle.index >= len(slot[].generations)
                    or slot[].generations[handle.index] != handle.generation
                    or handle.index not in slot[].values):
                raise Error("native state registry: state is not fitted or was released")
            return slot[].values[handle.index].copy()
        finally:
            shared_slot_unlock(lock)

    @staticmethod
    def release(handle: StateHandle) raises:
        var slot = Self.SLOT.get_or_create_ptr()
        if slot[].state_size != size_of[Self.T]():
            raise Error("native state registry: state layout differs")
        var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].lock.unsafe_ptr())
        var retired = Optional[ArcPointer[Self.T]]()
        shared_slot_lock(lock)
        try:
            if (handle.index < 0 or handle.index >= len(slot[].generations)
                    or slot[].generations[handle.index] != handle.generation
                    or handle.index not in slot[].values):
                raise Error("native state registry: state is not fitted or was released")
            retired = slot[].values.pop(handle.index)
            slot[].released.append(handle.index)
        finally:
            shared_slot_unlock(lock)
            # A final state's cleanup may enter another native provider.
            _ = retired^

    @staticmethod
    def replace(handle: StateHandle, var state: Self.T) raises:
        var fresh = ArcPointer[Self.T](state^)
        var retired = Optional[ArcPointer[Self.T]]()
        var slot = Self.SLOT.get_or_create_ptr()
        if slot[].state_size != size_of[Self.T]():
            raise Error("native state registry: state layout differs")
        var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](slot[].lock.unsafe_ptr())
        shared_slot_lock(lock)
        try:
            if (handle.index < 0 or handle.index >= len(slot[].generations)
                    or slot[].generations[handle.index] != handle.generation
                    or handle.index not in slot[].values):
                raise Error("native state registry: state is not fitted or was released")
            retired = slot[].values.pop(handle.index)
            slot[].values[handle.index] = fresh.copy()
        finally:
            shared_slot_unlock(lock)
            _ = retired^
            _ = fresh
