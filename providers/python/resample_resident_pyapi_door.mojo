# SPDX-License-Identifier: Apache-2.0
"""UNPACK of the `resident()` scope for the public resample functions (lane gd-rf-resample, 2026-11-25).

Shared by bindings/linalg_ops_resample_door.mojo (bootstrap, permutation_test) and
providers/python/resample_rows_pyapi_door.mojo (resample). No numerics.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_get
from bindings.native_loaded_module_door import native_loaded_module
from bindings.admitted_input_unpack import AdmittedBuf
from bindings.arena_cache_run import ArenaCachePut, arena_scopes_live, arena_scope_floor, arena_scope_store_put
from bindings.arena_scope_door import arena_scope_active
from bindings.arena_owner_door import arena_pin, arena_unpin
from bindings.metric_resident_source_door import metric_resident_owner
from bindings.classical_arrays_door import classical_address
from core.entry_table import EntryRef, entry_ref_word, entry_ref_unresolved
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_connect_factory_once, neural_descriptor_entry,
    neural_descriptor_execute,
)
from x_prep.transport_abi import XP_TRANSPORT_ABI, XP_OP_COUNT

#: The table operation (resample/resample_native.mojo op 7): the image's resident store route word, -1 none
#: (resample/resample_store.mojo `resample_store_route`; the host image answers -1).
comptime RS_OP_STORE_ROUTE = 7

# The `resident()` scope (lane gd-rf-resample, 2026-11-25). Inside
# `with mojolearn._arena_io.resident():` a GPU image keeps each input array
# ONCE in its own device store (resample/resample_store.mojo, the capsule
# `resample_store_v1`, x_prep transport numbering), keyed by the CALLER'S
# buffer (bindings/metric_resident_source_door.mojo `metric_resident_owner`:
# an admitted copy of the same words is keyed by the caller's object, so a
# later call on the same buffer finds it), and the operation reads the slot
# in place (its upload site: resample/estimator.mojo `_resample_input`).
# Outside every scope `rs_resident` is one atomic load and nothing else runs:
# the ints are the plain table's and each input is uploaded as before.

#: The scope cache's family prefix of the resample store
#: (resample/resample_store.mojo RESAMPLE_STORE_PREFIX).
comptime _RS_STORE_PREFIX = "resample"


struct RsResident(Movable):
    """The run's `resident()` scope on the resample image's store, or none.
    `store_pin` keeps the store's connected provider alive for the run."""
    var scope: Int
    var floor: Int
    var store: EntryRef
    var store_pin: PythonObject

    def __init__(out self):
        self.scope = -1
        self.floor = 0
        self.store = entry_ref_unresolved()
        self.store_pin = PythonObject(None)

    def active(self) -> Bool:
        return self.scope >= 0


def rs_resident(binding: PythonObject, provider: PythonObject, context: PythonObject) raises -> RsResident:
    """The calling thread's active scope on `binding`'s store. None (inactive)
    when no scope is alive in the process (one atomic load: the only work
    outside a scope), when the thread is outside one, or when the image has
    no store (the host image answers -1 to RS_OP_STORE_ROUTE). Once per run."""
    var rs = RsResident()
    if not arena_scopes_live():
        return rs^
    var local = py_get(context, "arena_local")
    if local is None:
        return rs^
    var scope = arena_scope_active(local, native_loaded_module("_thread")._local,
                                   native_loaded_module("builtins").str, context["arena_scope_owner"],
                                   context["arena_scope_class"])
    if scope < 0:
        return rs^
    var route = neural_descriptor_execute(provider, RS_OP_STORE_ROUTE, List[Int](), List[Int](), List[Float64](),
                                          False)
    if route < 0:
        return rs^
    var factory = neural_descriptor_entry(binding, "resample_store_v1", False)
    if factory is None:
        return rs^
    var store_provider = neural_descriptor_connect_factory_once(factory, String(XP_TRANSPORT_ABI), XP_OP_COUNT)
    var entry = store_provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry
    rs.store = EntryRef(entry, entry_ref_word((route >> 8) & 0xFF, route & 0xFF, XP_OP_COUNT))
    rs.store_pin = store_provider
    rs.floor = arena_scope_floor(scope)
    rs.scope = scope
    return rs^


def rs_resident_slot(rs: RsResident, source: PythonObject, admitted: AdmittedBuf) raises -> Int:
    """The store slot holding the float32 words of `admitted` (admitted from
    the public argument `source`), kept once by the scope and found again by
    every later call on the same caller buffer; -1 outside a scope, below the
    scope's size floor, or for a converted per-call copy (never kept: its
    address is new each call). The cache pins the caller's object."""
    if not rs.active():
        return -1
    var words = admitted.count()
    if words < 1 or words < rs.floor:
        return -1
    var owner = metric_resident_owner(source, admitted.keep)
    if owner is None:
        return -1
    var address = admitted.addr if owner is admitted.keep else Int(py=classical_address(owner))
    return rs_resident_slot_at(rs, owner, address, words)


def rs_resident_slot_at(rs: RsResident, owner: PythonObject, address: Int, words: Int) raises -> Int:
    """The store slot of the caller's buffer (`owner`, its data at `address`,
    `words` 4-byte words), kept once by the scope (the cache pins `owner`)
    and found again by every later call on it; -1 outside a scope or below
    the scope's size floor. The caller has checked that `address` is
    `owner`'s own buffer (never a per-call copy)."""
    if not rs.active() or words < 1 or words < rs.floor:
        return -1
    var pin = arena_pin(owner)
    var put = ArenaCachePut(-1, False)
    try:
        with GILReleased(Python()):
            put = arena_scope_store_put(rs.scope, rs.store, String(_RS_STORE_PREFIX), address, words, pin)
    except error:
        arena_unpin(pin)
        raise error^
    if not put.pinned:
        arena_unpin(pin)  # found (a hit), or another thread published it first
    return put.store_id
