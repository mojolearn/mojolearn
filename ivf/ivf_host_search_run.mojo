# SPDX-License-Identifier: Apache-2.0
"""RUN bodies of the IVF-Flat host search and extend (bindings/ivf_host_search.mojo):
the index built from admitted arrays, the search, the extend, and the
resident registry with its lock word. Typed Mojo only, no Python import
(tools/python_free_dirs.sh). The doors keep the list-count checks, the
extent admission, the reads of the caller's address words and the GIL
release around each search exactly where it was; the statements here are the
former door statements, unchanged, so the bits are the same (lane
rs-decomposition, 2026-11-12)."""
from std.atomic import Atomic, Ordering
from std.memory import ArcPointer

from core.hostptr import read_f32
from ivf.host.ivf_host import IvfHostIndex, host_ivf_extend, host_ivf_search
from ivf.ivf_index_arrays_run import IvfIndexArrays, ivf_write_extended_arrays_run
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.shared_slot import SharedSlot


def ivf_host_index_of(var arrays: IvfIndexArrays) raises -> IvfHostIndex:
    """The host index over an admitted index's lists, moved (no copy)."""
    # compile-fix-1: each list moves out and its field is refilled empty, so
    # `arrays` stays whole and is destroyed as one value (moving fields out of
    # the middle of it does not compile); still no copy of the index arrays.
    var taken_centers = arrays.centers^
    arrays.centers = List[Float32]()
    var taken_center_norms = arrays.center_norms^
    arrays.center_norms = List[Float32]()
    var taken_offsets = arrays.offsets^
    arrays.offsets = List[Int32]()
    var taken_list_indices = arrays.list_indices^
    arrays.list_indices = List[UInt32]()
    var taken_list_data = arrays.list_data^
    arrays.list_data = List[Float32]()
    var index = IvfHostIndex(
        arrays.n_lists, arrays.dim, arrays.n_rows, arrays.metric,
        taken_centers^, taken_center_norms^, taken_offsets^,
        taken_list_indices^, taken_list_data^,
    )
    return index^


def ivf_host_search_run(
    index: IvfHostIndex, queries: List[Float32], m: Int, k: Int, n_probes: Int, partial_storage: Bool,
    keep: List[Int32], mut dist: List[Float32], mut idx: List[UInt32], mut cand: List[Int32],
) raises:
    """`host_ivf_search` with its three result lists moved out; the door
    calls it with the GIL released."""
    var r = host_ivf_search(index, queries, m, k, n_probes, partial_storage, keep)
    # compile-fix-2: each result list moves out and its field is refilled
    # empty, so `r` stays whole and is destroyed as one value (moving
    # fields out of the middle of it does not compile); still no copy.
    dist = r.distances^
    r.distances = List[Float32]()
    idx = r.indices^
    r.indices = List[UInt32]()
    cand = r.n_candidates^
    r.n_candidates = List[Int32]()


def ivf_host_extend_run(
    var arrays: IvfIndexArrays, new_x_addr: Int, n_new: Int, offsets_addr: Int, indices_addr: Int,
    list_data_addr: Int, labels_addr: Int,
) raises:
    """`host_ivf_extend` over an admitted index, the extended lists written
    to the four output buffers (`ivf_flat_extend_binding`)."""
    var new_x = read_f32(new_x_addr, n_new * arrays.dim)
    var index = ivf_host_index_of(arrays^)
    var labels = List[UInt32]()
    var out = host_ivf_extend(index, new_x, n_new, labels)
    ivf_write_extended_arrays_run(
        offsets_addr, indices_addr, list_data_addr, labels_addr, out.n_rows, out.dim, out.n_lists,
        out.offsets, out.list_indices, out.list_data, labels, n_new,
    )


struct HostIvfResident(Movable):
    var index: IvfHostIndex
    var partial_storage: Bool

    def __init__(out self, var index: IvfHostIndex, partial_storage: Bool):
        self.index = index^
        self.partial_storage = partial_storage


struct HostIvfRegistry(Defaultable, Movable):
    # p3-neighbors-cluster: each entry sits behind a reference-counted
    # pointer. A search copies the pointer under the registry's lock word and
    # reads the index with the GIL released; a concurrent prepare (which may
    # grow the dictionary) or release then drops only the registry's
    # reference, never the memory a running search reads.
    var entries: Dict[Int, ArcPointer[HostIvfResident]]
    var next_id: Int
    # One lock word (the neighbors/resident_index.mojo pattern): searches run
    # with the GIL released.
    var lock: List[Int64]

    def __init__(out self):
        self.entries = Dict[Int, ArcPointer[HostIvfResident]]()
        self.next_id = 1
        self.lock = List[Int64](length=1, fill=Int64(0))


# V2 -> V3: the entry layout changed (p3-neighbors-cluster); a process-wide
# slot name is shared by every extension that registers it.
comptime HOST_IVF_REGISTRY = SharedSlot[HostIvfRegistry, (
    "MojolearnIvfFlatHostResidentIdenticalV3" if GLOBAL_NUMERIC_MODE == 1 else
    "MojolearnIvfFlatHostResidentFastV3"
)]


def _host_ivf_registry() raises -> UnsafePointer[HostIvfRegistry, MutAnyOrigin]:
    return HOST_IVF_REGISTRY.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _host_ivf_lock(state: UnsafePointer[HostIvfRegistry, MutAnyOrigin]) raises:
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the registry's lock word (held for one dictionary step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _host_ivf_unlock(state: UnsafePointer[HostIvfRegistry, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def _host_ivf_view(h: Int) raises -> ArcPointer[HostIvfResident]:
    """The entry's pointer (a reference-counted copy), read under the lock
    word."""
    var state = _host_ivf_registry()
    _host_ivf_lock(state)
    try:
        if h not in state[].entries:
            raise Error("unknown or released resident IVF-Flat index handle")
        var view = state[].entries[h].copy()
        _host_ivf_unlock(state)
        return view^
    except error:
        _host_ivf_unlock(state)
        raise error^


def ivf_host_register(var arrays: IvfIndexArrays, partial: Bool) raises -> Int:
    """The resident host index from an admitted index; its handle
    (`ivf_flat_index_prepare_binding`)."""
    var index = ivf_host_index_of(arrays^)
    var entry = ArcPointer[HostIvfResident](HostIvfResident(index^, partial))
    var state = _host_ivf_registry()
    _host_ivf_lock(state)
    try:
        if state[].next_id == 9223372036854775807:
            raise Error("resident IVF-Flat index handle space exhausted")
        var handle = state[].next_id
        state[].next_id += 1
        state[].entries[handle] = entry^
        _host_ivf_unlock(state)
        return handle
    except error:
        _host_ivf_unlock(state)
        raise error^


def ivf_host_release(h: Int) raises:
    """Drop the registry's reference to the resident host index `h`
    (`ivf_flat_index_release_binding`)."""
    var state = _host_ivf_registry()
    _host_ivf_lock(state)
    if h not in state[].entries:
        _host_ivf_unlock(state)
        raise Error("unknown or released resident IVF-Flat index handle")
    try:
        var released = state[].entries.pop(h)
        _host_ivf_unlock(state)
        # The registry's reference is dropped outside the lock; a running
        # search keeps its own copy until it returns.
        _ = released^
    except error:
        _host_ivf_unlock(state)
        raise error^
