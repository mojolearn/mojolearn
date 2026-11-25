# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The process-wide handle tables of the DART rounds (lane gd-dart-threads,
2026-11-25): the host sessions (xtrees/dart_host.mojo), the device sessions
and the kept device forests (xtrees/dart_device.mojo).

THE TABLE IS READ THROUGH A POINTER, AFTER THE LOCK. The three registries
used to be structs whose `mint`, `keep`, `lease` and `retire` methods took
the lock and then read `self.next_id` / `self.sessions`. A field read
through a method's `self` argument may be loaded once, before the lock's
compare-and-swap, and kept (the defect core/forest_export_run.mojo and
core/msel_host.mojo were fixed for on 2026-11-22): two threads opening a
session at once could read the same `next_id` and both mint it. The second
member fit then leased the FIRST session with that id, another fold's, and
refused `rf_regressor_fit_dart: params name a shape the DART session does
not hold` (proving test_two_threads_cross_val_score_like_one
[DARTRegressor-cpu], L40S 2026-11-25). Here every operation is a static
function: it takes the slot's pointer as a local, locks, and reads and
writes the table through that pointer, so every read is ordered after the
acquire. The id is minted and the entry kept in ONE locked step (`add`), so
no id is ever visible without its entry. Nothing else changed: the same
lock word, the same lookup over the live handles, the entry held by a copy
of its `ArcPointer` so the work runs with no lock (lane fx-gil).

Not a hot path: one lookup per DART open, round step, member fit, add and
close (a few live handles), as before."""
from std.memory import ArcPointer
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock


struct DartHandleTable[T: Movable & Deinitable](Defaultable, Movable):
    """The live entries and their ids (`ids[i]` names `entries[i]`), the
    next id, and `lock` (word 0) guarding all three."""

    var lock: List[Int64]
    var ids: List[Int]
    var entries: List[ArcPointer[Self.T]]
    var next_id: Int

    def __init__(out self):
        # side-effect free (core/shared_slot.mojo, FIRST CREATION)
        self.lock = List[Int64](length=1, fill=Int64(0))
        self.ids = List[Int]()
        self.entries = List[ArcPointer[Self.T]]()
        self.next_id = 1


struct DartHandles[T: Movable & Deinitable, name: StaticString, version: Int, refusal: StaticString]:
    """A process-wide DART handle table: `add` keeps an entry under a new
    id, `lease` hands out a held copy, `retire` takes it out (it is freed
    when its last holder lets go). `refusal` is the wording of an unknown
    id. Bump the slot name's `V<n>` and `version` together on any layout
    change of `T` or of the table."""

    comptime SLOT = SharedSlot[DartHandleTable[Self.T], Self.name, Self.version]

    @staticmethod
    def add(var entry: Self.T) raises -> Int:
        """Keep `entry` under a new id, minted in the same locked step."""
        # the allocation runs before the lock
        var held = ArcPointer[Self.T](entry^)
        var table = Self.SLOT.get_or_create_ptr()
        var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](table[].lock.unsafe_ptr())
        shared_slot_lock(word)
        var id = table[].next_id
        table[].next_id += 1
        table[].ids.append(id)
        table[].entries.append(held^)
        shared_slot_unlock(word)
        return id

    @staticmethod
    def lease(id: Int) raises -> ArcPointer[Self.T]:
        """The entry of `id`, held by the returned copy."""
        var table = Self.SLOT.get_or_create_ptr()
        var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](table[].lock.unsafe_ptr())
        shared_slot_lock(word)
        try:
            for i in range(len(table[].ids)):  # small-loop(live handles): a handle lookup over live fits
                if table[].ids[i] == id:
                    return table[].entries[i].copy()
            raise Error(String(Self.refusal))
        finally:
            shared_slot_unlock(word)

    @staticmethod
    def retire(id: Int) raises -> ArcPointer[Self.T]:
        """The entry leaves the table; it is freed when the caller and every
        operation still holding it let go."""
        var table = Self.SLOT.get_or_create_ptr()
        var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](table[].lock.unsafe_ptr())
        shared_slot_lock(word)
        try:
            for i in range(len(table[].ids)):  # small-loop(live handles): a handle lookup over live fits
                if table[].ids[i] == id:
                    _ = table[].ids.pop(i)
                    return table[].entries.pop(i)
            raise Error(String(Self.refusal))
        finally:
            shared_slot_unlock(word)
