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
thread, aside)."""
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


struct ForestExportRegistry[Model: Movable & Deinitable](Defaultable, Movable):
    var entries: Dict[Int, ForestExportEntry[Self.Model]]
    var next_id: Int
    # One lock word (module note): heap storage, so its address is stable.
    var lock: List[Int64]
    # The handles `take` handed out and `put_back` has not returned yet
    # (module note, A SECOND READER WAITS). Read and written under the lock.
    var out_ids: List[Int]

    def __init__(out self):
        self.entries = Dict[Int, ForestExportEntry[Self.Model]]()
        self.next_id = 1
        self.lock = List[Int64](length=1, fill=Int64(0))
        self.out_ids = List[Int]()

    def _lock(self):
        var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](self.lock.unsafe_ptr())
        while True:  # driver-loop: spin on the registry's lock word (held for one dictionary step only)
            var expected = Int64(0)
            if Atomic.compare_exchange[
                success_ordering=Ordering.SEQUENTIAL,
                failure_ordering=Ordering.RELAXED,
            ](word, expected, Int64(1)):
                return

    def _unlock(self):
        var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](self.lock.unsafe_ptr())
        Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))

    def _hold(self, handle: Int) raises:
        """Return WITH THE LOCK HELD and `handle` kept in `entries`. A handle
        another thread has taken out is waited for (lock dropped, the CPU
        given away, looked at again); a handle neither kept nor out raises
        with the lock dropped."""
        while True:  # driver-loop: wait for another thread's put back of this handle (one export copy)
            self._lock()
            if handle in self.entries:
                return
            var busy = False
            for i in range(len(self.out_ids)):  # small-loop(out: the handles being read right now, at most one per thread)
                if self.out_ids[i] == handle:
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
        self._lock()
        if self.next_id == 9223372036854775807:
            self._unlock()
            raise Error("fitted forest export handle space exhausted")
        var handle = self.next_id
        self.next_id += 1
        self.entries[handle] = entry^
        self._unlock()
        return handle

    def take(mut self, handle: Int) raises -> ForestExportEntry[Self.Model]:
        """The kept entry, moved OUT of the registry (module note, THE LOCK):
        the caller reads it with no lock held and gives it back with
        `put_back` on every path."""
        self._hold(handle)
        var entry = self.entries.pop(handle)
        self.out_ids.append(handle)
        self._unlock()
        return entry^

    def _take_last(mut self, handle: Int) raises -> ForestExportEntry[Self.Model]:
        """`take` for the release: the entry leaves for good, so its handle
        is not listed as out (a later asker is told it is released)."""
        self._hold(handle)
        var entry = self.entries.pop(handle)
        self._unlock()
        return entry^

    def put_back(mut self, handle: Int, var entry: ForestExportEntry[Self.Model]):
        """An entry `take` handed out, kept again under its handle."""
        self._lock()
        self.entries[handle] = entry^
        var last = len(self.out_ids) - 1
        for i in range(last + 1):  # small-loop(out: the handles being read right now)
            if self.out_ids[i] == handle:
                self.out_ids[i] = self.out_ids[last]
                _ = self.out_ids.pop()
                break
        self._unlock()

    def sizes_into(self, handle: Int, out_addr: Int) raises:
        """XF_EXPORT_SIZES (core/forest_fit_request.mojo)."""
        var trees = 0
        var nodes = 0
        var outputs = 0
        var meta = List[Int64]()
        # Returns with the lock held and the handle kept; an unknown handle
        # raises, a handle out with another thread is waited for.
        self._hold(handle)
        ref entry = self.entries[handle]
        trees = entry.trees
        nodes = entry.nodes
        outputs = entry.outputs
        meta = entry.meta.copy()
        self._unlock()
        forest_export_sizes_into(out_addr, trees, nodes, outputs, meta)

    def meta_of(self, handle: Int) raises -> List[Int64]:
        var meta = List[Int64]()
        # Returns with the lock held and the handle kept; an unknown handle
        # raises, a handle out with another thread is waited for.
        self._hold(handle)
        meta = self.entries[handle].meta.copy()
        self._unlock()
        return meta^

    def shape_of(self, handle: Int) raises -> List[Int]:
        """[trees, nodes, outputs] of the kept entry (the Python descriptor)."""
        var shape = List[Int](length=3, fill=0)
        # Returns with the lock held and the handle kept; an unknown handle
        # raises, a handle out with another thread is waited for.
        self._hold(handle)
        ref entry = self.entries[handle]
        shape[0] = entry.trees
        shape[1] = entry.nodes
        shape[2] = entry.outputs
        self._unlock()
        return shape^

    def validate(self, handle: Int, trees: Int, nodes: Int, outputs: Int) raises:
        var same = False
        # Returns with the lock held and the handle kept; an unknown handle
        # raises, a handle out with another thread is waited for.
        self._hold(handle)
        ref entry = self.entries[handle]
        same = entry.trees == trees and entry.nodes == nodes and entry.outputs == outputs
        self._unlock()
        if not same:
            raise Error("fitted forest export destination sizes differ from owned model")

    def release(mut self, handle: Int) raises:
        # The model is destroyed after the lock is dropped.
        var released = self._take_last(handle)
        _ = released^
