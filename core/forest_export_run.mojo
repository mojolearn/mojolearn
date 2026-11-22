# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/forest_export_binding.mojo` (wave 8 run split): the typed
fitted-forest export registry (insert, sizes, validate, release). Imports no
Python; the binding keeps the Python descriptor (`forest_export_insert_py`,
`forest_export_descriptor_py`).

THE LOCK (lane fx-gil, 2026-11-25; THE GIL RULE, core/member_abi.mojo). The
registry was serialized by the GIL. A member's fit on a worker holds no GIL,
so every read and write of `entries` and `next_id` takes the registry's own
lock word, held for ONE dictionary step (mint, lookup, pop, put back) and
never across a copy of a model, a kernel launch or a wait for the GIL. A
reader of a kept model TAKES the entry out (`take`), reads it with no lock
held and PUTS IT BACK (`put_back`): a `ref` into `entries` would dangle when
another thread's insert rehashes the dictionary.

A SECOND READER WAITS, IT IS NEVER REFUSED (round 4). While an entry is out
its handle is listed in `out_ids`. Every operation on that handle from
another thread (a second export or save of the same fitted forest, its sizes,
its release) finds the handle there and WAITS for the put back (`_hold`:
drop the lock, yield, look again); only a handle that is neither kept nor out
is "unknown or released". The wait cannot deadlock on the GIL: between `take`
and `put_back` a holder never changes its GIL state (it holds the GIL for the
whole read, or holds none and asks for none), so a waiter that holds the GIL
is never what the holder needs. A new reader keeps that property.
Nobody reads `entries` directly outside this file (the protocol check, one
thread, aside).

THE TABLE IS BEHIND A POINTER (lane pv-race, 2026-11-22). The dictionary, the
next handle and the out list live in `_ExportTable`, one element of the
heap List `cell`, and every method reaches them through `_table()` (a
pointer made from that heap address) after its `_lock`. They used to be
fields of the registry itself, read through the method's `self`. A method
that takes `self` immutably (`_hold`, `sizes_into`, `validate`, `meta_of`,
`shape_of`) hands the compiler a read-only, non-aliased argument: LLVM may
then treat that memory as unchanging for the whole call, and it did. The
45e90deaa7-era `_hold` (objdump of _mojolearn_native_forest_host.so, ~/mojolearn-evidence/briefs-20261122/pv-race/)
loaded the Dict's capacity and `len(out_ids)` once, at entry, BEFORE its
first `lock cmpxchg`, and looped on that snapshot; `validate` / `sizes_into`
read `entries[handle]` with internals taken before the lock. With member
fits on parallel lanes (MEMBER_GIL_FREE) another lane's insert rehashed the
Dict between the snapshot and the lock: a read of freed index storage, a
list copy of a garbage length ("alloc: Layout.count() must be >= 0") and the
corrupted Dict in `put_back` / `_retain_rf_handle` (SIGSEGV / SIGILL; lane
sr-other's crash_meta.py). Memory reached through a pointer the method
loads is not the argument's: the acquire compare-and-swap orders every read
of it. A new field of shared state goes into `_ExportTable`, never into the
registry struct."""
from std.atomic import Atomic, Ordering
from std.ffi import c_int, external_call
from core.forest_fit_request import forest_export_sizes_into
from core.forest_export_binding_impl import validate_forest_export_counts


@fieldwise_init
struct ForestExportEntry[Model: Movable & Deinitable](Movable):
    var model: Self.Model
    var trees: Int
    var nodes: Int
    var outputs: Int
    var meta: List[Int64]


struct _ExportTable[Model: Movable & Deinitable](Movable):
    """The registry's shared state, in heap storage of its own (module note,
    THE TABLE IS BEHIND A POINTER)."""

    var entries: Dict[Int, ForestExportEntry[Self.Model]]
    var next_id: Int
    # The handles `take` handed out and `put_back` has not returned yet
    # (module note, A SECOND READER WAITS). Read and written under the lock.
    var out_ids: List[Int]

    def __init__(out self):
        self.entries = Dict[Int, ForestExportEntry[Self.Model]]()
        self.next_id = 1
        self.out_ids = List[Int]()


struct ForestExportRegistry[Model: Movable & Deinitable](Defaultable, Movable):
    # One lock word (module note): heap storage, so its address is stable.
    var lock: List[Int64]
    # Exactly one `_ExportTable`, never grown, so its address is stable too;
    # every read and write goes through `_table()` with the lock held.
    var cell: List[_ExportTable[Self.Model]]

    def __init__(out self):
        self.lock = List[Int64](length=1, fill=Int64(0))
        self.cell = List[_ExportTable[Self.Model]]()
        self.cell.append(_ExportTable[Self.Model]())

    def _table(self) -> MutPointer[_ExportTable[Self.Model], MutUntrackedOrigin]:
        return MutPointer[_ExportTable[Self.Model], MutUntrackedOrigin](
            unsafe_from_address=Int(self.cell.unsafe_ptr())
        )

    def _lock(self):
        var word = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(self.lock.unsafe_ptr()))
        while True:  # driver-loop: spin on the registry's lock word (held for one dictionary step only)
            var expected = Int64(0)
            if Atomic.compare_exchange[
                success_ordering=Ordering.SEQUENTIAL,
                failure_ordering=Ordering.RELAXED,
            ](word, expected, Int64(1)):
                return

    def _unlock(self):
        var word = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(self.lock.unsafe_ptr()))
        Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))

    def _hold(self, handle: Int) raises:
        """Return WITH THE LOCK HELD and `handle` kept in the table. A handle
        another thread has taken out is waited for (lock dropped, the CPU
        given away, looked at again); a handle neither kept nor out raises
        with the lock dropped."""
        var table = self._table()
        while True:  # driver-loop: wait for another thread's put back of this handle (one export copy)
            self._lock()
            if handle in table[].entries:
                return
            var busy = False
            for i in range(len(table[].out_ids)):  # small-loop(out: the handles being read right now, at most one per thread)
                if table[].out_ids[i] == handle:
                    busy = True
            self._unlock()
            if not busy:
                raise Error("unknown or released fitted forest export handle")
            _ = external_call["sched_yield", c_int]()

    def insert_native(mut self, var model: Self.Model, trees: Int, nodes: Int,
                      outputs: Int, var meta: List[Int64]) raises -> Int:
        """Move one fitted native result into ownership; return its handle
        (the forest entry's fits, lane calltable-forest)."""
        validate_forest_export_counts(trees, nodes, outputs)
        var entry = ForestExportEntry[Self.Model](model^, trees, nodes, outputs, meta^)
        var table = self._table()
        self._lock()
        if table[].next_id == 9223372036854775807:
            self._unlock()
            raise Error("fitted forest export handle space exhausted")
        var handle = table[].next_id
        table[].next_id += 1
        table[].entries[handle] = entry^
        self._unlock()
        return handle

    def take(mut self, handle: Int) raises -> ForestExportEntry[Self.Model]:
        """The kept entry, moved OUT of the registry (module note, THE LOCK):
        the caller reads it with no lock held and gives it back with
        `put_back` on every path."""
        var table = self._table()
        self._hold(handle)
        var entry = table[].entries.pop(handle)
        table[].out_ids.append(handle)
        self._unlock()
        return entry^

    def _take_last(mut self, handle: Int) raises -> ForestExportEntry[Self.Model]:
        """`take` for the release: the entry leaves for good, so its handle
        is not listed as out (a later asker is told it is released)."""
        var table = self._table()
        self._hold(handle)
        var entry = table[].entries.pop(handle)
        self._unlock()
        return entry^

    def put_back(mut self, handle: Int, var entry: ForestExportEntry[Self.Model]):
        """An entry `take` handed out, kept again under its handle."""
        var table = self._table()
        self._lock()
        table[].entries[handle] = entry^
        var last = len(table[].out_ids) - 1
        for i in range(last + 1):  # small-loop(out: the handles being read right now)
            if table[].out_ids[i] == handle:
                table[].out_ids[i] = table[].out_ids[last]
                _ = table[].out_ids.pop()
                break
        self._unlock()

    def sizes_into(self, handle: Int, out_addr: Int) raises:
        """XF_EXPORT_SIZES (core/forest_fit_request.mojo)."""
        var trees = 0
        var nodes = 0
        var outputs = 0
        var meta = List[Int64]()
        var table = self._table()
        # Returns with the lock held and the handle kept; an unknown handle
        # raises, a handle out with another thread is waited for.
        self._hold(handle)
        ref entry = table[].entries[handle]
        trees = entry.trees
        nodes = entry.nodes
        outputs = entry.outputs
        meta = entry.meta.copy()
        self._unlock()
        forest_export_sizes_into(out_addr, trees, nodes, outputs, meta)

    def meta_of(self, handle: Int) raises -> List[Int64]:
        var meta = List[Int64]()
        var table = self._table()
        # Returns with the lock held and the handle kept; an unknown handle
        # raises, a handle out with another thread is waited for.
        self._hold(handle)
        meta = table[].entries[handle].meta.copy()
        self._unlock()
        return meta^

    def shape_of(self, handle: Int) raises -> List[Int]:
        """[trees, nodes, outputs] of the kept entry (the Python descriptor)."""
        var shape = List[Int](length=3, fill=0)
        var table = self._table()
        # Returns with the lock held and the handle kept; an unknown handle
        # raises, a handle out with another thread is waited for.
        self._hold(handle)
        ref entry = table[].entries[handle]
        shape[0] = entry.trees
        shape[1] = entry.nodes
        shape[2] = entry.outputs
        self._unlock()
        return shape^

    def validate(self, handle: Int, trees: Int, nodes: Int, outputs: Int) raises:
        var same = False
        var table = self._table()
        # Returns with the lock held and the handle kept; an unknown handle
        # raises, a handle out with another thread is waited for.
        self._hold(handle)
        ref entry = table[].entries[handle]
        same = entry.trees == trees and entry.nodes == nodes and entry.outputs == outputs
        self._unlock()
        if not same:
            raise Error("fitted forest export destination sizes differ from owned model")

    def release(mut self, handle: Int) raises:
        # The model is destroyed after the lock is dropped.
        var released = self._take_last(handle)
        _ = released^
