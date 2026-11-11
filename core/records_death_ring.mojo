# SPDX-License-Identifier: Apache-2.0
"""The dead-key ring of the native model records: released model identity
words, kept for polling by the family stores keyed by model identity.

`bindings/native_model_records.mojo` posts a key here when a model's record
is released; a store (the sequence resident store, the cluster/spectral
resident table) keeps the last sequence word it drained and pulls the keys
released since. Host-only, standard library only: no Python and no bindings
import, so kernel modules can poll the ring without reaching the binding
shell (the kernel-closure rule: bindings import kernels, never the reverse).
"""
from std.atomic import Atomic, Ordering
from core.shared_slot import SharedSlot


# Moved out of bindings/native_model_records.mojo (2026-11-07, lane
# post-kernel-imports-shell) unchanged, so the sequence kernel's import does
# not reach the shell; the _Global name is the same, so the ring is the same
# process-wide object.
#
# Replaces the release-hook registry (thin function pointers in a _Global,
# the trigger of the compiler deadlock bisected to the cluster resident
# table's registration, lanes ne-fix-hang4/5, 2026-11-07): no function
# pointer, no cross-module callback, no cycle. Every released key is written
# to a fixed ring with a monotonically increasing sequence word; a store
# keeps the last sequence it drained and polls at its own entry points.

comptime RECORDS_DEAD_RING = 1024
"""Released keys kept for polling. A store that has not drained for more
releases than this forgets everything it holds (its next call re-uploads):
a bound on host memory (8 KiB), not a shape rule."""


struct RecordsDeathRing(Defaultable, Movable):
    var keys: Array[Int, RECORDS_DEAD_RING]
    # Releases posted so far; the release numbered s (1-based) sits at
    # keys[(s - 1) % RECORDS_DEAD_RING].
    var seq: Int
    # One lock word: the post runs with the GIL held (a weakref callback),
    # a poll may run from a device entry with the GIL released.
    var lock: List[Int64]

    def __init__(out self):
        self.keys = Array[Int, RECORDS_DEAD_RING](fill=0)
        self.seq = 0
        self.lock = List[Int64](length=1, fill=Int64(0))


# Shared by every Mojo extension in the process (one ring, like the table).
comptime MODEL_RECORDS_DEATHS = SharedSlot[RecordsDeathRing, "MojolearnModelRecordsDeathsV2"]


def _deaths() raises -> UnsafePointer[RecordsDeathRing, MutAnyOrigin]:
    return MODEL_RECORDS_DEATHS.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _deaths_lock(ring: UnsafePointer[RecordsDeathRing, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](ring[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the ring's lock word (held for a few word copies only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _deaths_unlock(ring: UnsafePointer[RecordsDeathRing, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](ring[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def records_post_death(model_key: Int):
    """Append `model_key` to the ring (overwriting the oldest when full).
    Non-raising: it runs from death callbacks."""
    try:
        var ring = _deaths()
        _deaths_lock(ring)
        ring[].keys[ring[].seq % RECORDS_DEAD_RING] = model_key
        ring[].seq += 1
        _deaths_unlock(ring)
    except:
        pass


def records_death_seq() -> Int:
    """The number of releases posted so far (a store's starting point)."""
    try:
        var ring = _deaths()
        _deaths_lock(ring)
        var seq = ring[].seq
        _deaths_unlock(ring)
        return seq
    except:
        return 0


def records_dead_keys_since(seq: Int, mut keys: List[Int]) -> Int:
    """Append to `keys` every key released after sequence word `seq` (oldest
    first) and return the current sequence word, the caller's next `seq`.
    When more keys were released since `seq` than the ring holds, nothing is
    appended and the NEGATED current sequence word is returned (always < 0):
    the caller forgets everything it holds and keeps `-result` as its next
    `seq`. Non-raising; any thread (the ring has its own lock word)."""
    try:
        var ring = _deaths()
        _deaths_lock(ring)
        var now = ring[].seq
        if seq >= now:
            _deaths_unlock(ring)
            return now
        if seq < 0 or now - seq > RECORDS_DEAD_RING:
            _deaths_unlock(ring)
            return -now
        for s in range(seq, now):  # driver-loop: over releases posted since the caller's last poll (at most RECORDS_DEAD_RING)
            keys.append(ring[].keys[s % RECORDS_DEAD_RING])
        _deaths_unlock(ring)
        return now
    except:
        return seq
