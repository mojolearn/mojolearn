# SPDX-License-Identifier: Apache-2.0
"""The pending slot of the resident IVF-PQ fit (lane dw-neighbors,
2026-11-21): where a built index waits between the two operations of the ANN
operations table (core/neighbors_entry_table_ann.mojo) that the resident fit
is made of.

`ANN_OP_IVF_PQ_BUILD_RESIDENT` runs with the GIL released and ends in
`ann_pending_park`: the built host index and the device buffers the build
left behind, kept under a ticket. `ANN_OP_IVF_PQ_REGISTER_BUILT` runs with
the GIL held and starts in `ann_pending_take`: both values back, to be
registered in the resident registry (x_ann/resident.mojo), which is guarded
by the GIL and has no lock word. The two operations are the two phases the
`x_ann_ivf_pq_build_resident` export always had (build released, register
held); the slot only carries the values across the table call that now sits
between them, because the fit door is another binary.

The slot is reached without the GIL (two threads may fit at once), so it has
its own lock word (the ivf/ivf_host_search_run.mojo pattern), held for one
dictionary step: three uncontended acquisitions per resident fit (park,
take the index, take the buffers), none per query. The index and the device buffers sit in two dictionaries and are
popped whole, so no field is moved out of an aggregate.

Serial by nature: one dictionary step per fit; there is nothing to split.
"""
from std.atomic import Atomic, Ordering
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.shared_slot import SharedSlot
from x_ann.ivf_pq_core import IvfPqIndex
from x_ann.ivf_pq_device import IvfPqDevice


struct AnnPendingPq(Defaultable, Movable):
    var built: Dict[Int, IvfPqIndex]
    var devices: Dict[Int, IvfPqDevice]
    var next_id: Int
    var lock: List[Int64]

    def __init__(out self):
        self.built = Dict[Int, IvfPqIndex]()
        self.devices = Dict[Int, IvfPqDevice]()
        self.next_id = 1
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime ANN_PENDING_PQ = SharedSlot[AnnPendingPq, (
    "MojolearnXAnnPendingPqIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnXAnnPendingPqFastV1"
)]


def _pending() raises -> UnsafePointer[AnnPendingPq, MutAnyOrigin]:
    return ANN_PENDING_PQ.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _pending_lock(state: UnsafePointer[AnnPendingPq, MutAnyOrigin]) raises:
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the slot's lock word (held for one dictionary step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _pending_unlock(state: UnsafePointer[AnnPendingPq, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def ann_pending_park(var built: IvfPqIndex, var dev: IvfPqDevice) raises -> Int:
    """Keep a built resident IVF-PQ index until it is registered; its
    ticket (from 1; 0 is the table's "no resident fit" answer)."""
    var state = _pending()
    _pending_lock(state)
    try:
        if state[].next_id == 9223372036854775807:
            raise Error("resident ann index ticket space exhausted")
        var ticket = state[].next_id
        state[].next_id += 1
        state[].built[ticket] = built^
        state[].devices[ticket] = dev^
        _pending_unlock(state)
        return ticket
    except error:
        _pending_unlock(state)
        raise error^


def ann_pending_take_built(ticket: Int) raises -> IvfPqIndex:
    """The host index parked under `ticket`, removed from the slot. Call
    `ann_pending_take_device` for the same ticket right after."""
    var state = _pending()
    _pending_lock(state)
    if ticket not in state[].built:
        _pending_unlock(state)
        raise Error("unknown or already registered resident IVF-PQ build ticket")
    try:
        var built = state[].built.pop(ticket)
        _pending_unlock(state)
        return built^
    except error:
        _pending_unlock(state)
        raise error^


def ann_pending_take_device(ticket: Int) raises -> IvfPqDevice:
    """The device buffers parked under `ticket`, removed from the slot."""
    var state = _pending()
    _pending_lock(state)
    if ticket not in state[].devices:
        _pending_unlock(state)
        raise Error("unknown or already registered resident IVF-PQ build ticket")
    try:
        var dev = state[].devices.pop(ticket)
        _pending_unlock(state)
        return dev^
    except error:
        _pending_unlock(state)
        raise error^
