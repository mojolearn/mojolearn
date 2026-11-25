# SPDX-License-Identifier: Apache-2.0
# The model-selection resident owner's entries (`_Resident.take`)
# and DeviceRows.materialize. They sit above both the Array preparation
# owner (arena_materialize_native) and the model-selection input owner
# (msel_inputs, which itself imports arena_materialize_native), so foreign
# containers reach the native materializer without an import cycle.
#
# Lane arena (2026-11-21): the entries are the native resident state
# (`core/arena_state.mojo` `ARENA_RESIDENTS`), found by the input's identity
# and published after the upload; the fold store is called through the
# owner's resolved helper entry (`core/msel_helpers_calls.mojo`) with the
# GIL released, as its Python exports did. The state lease is taken ONCE per
# public operation and passed down; nothing here runs per row.
from std.memory import ArcPointer
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call, py_type, py_get
from bindings.arena_materialize_native_door import (
    _prepare, _prepared_info, _cast, _slot, _empty, _empty_unfilled,
)
from bindings.arena_owner_door import (
    ArenaOwnerView, ArenaRows, arena_pin, arena_pinned, arena_unpin, arena_rows_publish_host,
)
from bindings.device_rows_door_run import DoorRowsStore
from bindings.msel_inputs_unpack import msel_input
from core.arena_state import ARENA_RESIDENTS, ArenaResidentState, ArenaResidentEntry, arena_handle
from core.msel_helpers_calls import msel_store_put, msel_store_free, msel_store_take_rows


@fieldwise_init
struct ResidentHit(Movable):
    """An input's resident entry and the admitted metadata of its prepared
    Array (admitted ONCE per call, p3-array-arena)."""
    var entry: ArenaResidentEntry
    var info: PythonObject


def _discard(entry: ArenaResidentEntry, view: ArenaOwnerView):
    # An upload that was not published: free its id and drop its pins. The
    # primary failure (or the entry another thread published) is what the
    # caller reports; a refused cleanup does not replace it.
    if entry.store_id >= 0:
        try:
            _ = msel_store_free(view.store, entry.store_id)
        except cleanup_error:
            pass
    arena_unpin(entry.owner_pin)
    arena_unpin(entry.prepared_pin)


def _publish(lease: ArcPointer[ArenaResidentState], entry: ArenaResidentEntry,
             view: ArenaOwnerView) raises -> ArenaResidentEntry:
    try:
        return lease[].publish(entry)
    except error:
        _discard(entry, view)
        raise error^


def _entry_info(values: PythonObject, lease: ArcPointer[ArenaResidentState], view: ArenaOwnerView,
                indices: Bool, context: PythonObject) raises -> ResidentHit:
    # The object's identity (CPython's id()) read from its native address:
    # calling builtins.id raises a `builtins.id` audit event per lookup. The
    # entry pins the object, so its address cannot be reused while it lives.
    var key = Int(values._obj_ptr)
    var found = lease[].find(key)
    if found.store_id != -2:
        var held = arena_pinned(found.prepared_pin)
        var held_info = _prepared_info(held, context)
        return ResidentHit(found^, held_info)
    var source = values
    if py_type(values) is not context["array_class"] and py_type(values) is not py_get(context,"ndarray"):
        # A foreign container (pandas, Arrow-backed pandas, list, tuple,
        # range, memoryview, array.array, torch CPU tensor) is materialized
        # once by the dtype-preserving model-selection input owner: native
        # admission, no __array__/DLPack/iteration hooks. Array and exact
        # ndarray stay on the zero-copy preparation below.
        var inputs = py_get(context,"arena_msel_input_context")
        if inputs is None:
            raise Error("RuntimeError: native resident rows require the model-selection input connection")
        source = msel_input(values,"fold data",inputs)
    var array = _prepare(source,context)
    if indices and String(_slot(array,"dtype",context)) != "<i8":
        array = _cast(array,"<i8",context)
    var prepared = _prepared_info(array,context)
    var nbytes = Int(py=prepared["nbytes"])
    var address = Int(py=prepared["address"])
    var shape = prepared["shape"]
    var rows = 0
    if len(shape):
        rows = Int(py=shape[0])
    var sid = -1
    if nbytes:
        with GILReleased(Python()):
            sid = msel_store_put(view.store, address, nbytes)
    var entry = ArenaResidentEntry(key, address, nbytes, rows, indices, sid, arena_pin(values), arena_pin(array))
    var kept = _publish(lease, entry, view)
    if kept.prepared_pin != entry.prepared_pin:
        # Another thread put the same input while the GIL was released for
        # this upload: its entry is the resident one, ours is freed.
        _discard(entry, view)
        var other = arena_pinned(kept.prepared_pin)
        var other_info = _prepared_info(other, context)
        return ResidentHit(kept^, other_info)
    _ = array
    return ResidentHit(kept^, prepared)


def arena_resident_take(view: ArenaOwnerView, values: PythonObject, indices_value: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    # _Resident.take: rows `indices_value` of `values`, gathered on the device
    # out of the resident store entries (each put once per owner).
    var lease = ARENA_RESIDENTS.acquire(arena_handle(view.word))
    var b = context["arena_native_types"]
    var base = _entry_info(values,lease,view,False,context)
    var index = _entry_info(indices_value,lease,view,True,context)
    var array = base.info
    var indices = index.info
    var shape = Python.list(indices["size"])
    var base_shape = array["shape"]
    for i in range(1,len(base_shape)):  # driver-loop: generic row shape preserves all trailing dimensions
        _ = py_call(shape.append,base_shape[i])
    # The take writes every byte of the output (indices.size x row_bytes,
    # one device download) or raises, discarding it: no zero fill first
    # (lane fb2-neighbors: a full memset of the train and test outputs,
    # n x d words per train_test_split, before the download overwrote them).
    # A source with no rows has nothing to take: its output is the zero-filled
    # Array (every byte written by the fill), as before the unfilled change
    # (fb6-smoke, 2026-11-17: replaces fb4-audit's refusal, which broke the
    # zero-size materialize contract). The fill runs only on that empty-source
    # corner, never on a take that downloads rows.
    var n = Int(py=base_shape[0])
    var output: PythonObject
    if n:
        output = _empty_unfilled(py_call(b["tuple"],shape),String(array["dtype"]),context)
    else:
        output = _empty(py_call(b["tuple"],shape),String(array["dtype"]),context)
    var row_bytes = Int(py=array["nbytes"])//n if n else 0
    var count = Int(py=indices["size"])
    if count and row_bytes:
        var dst = Int(py=_slot(output,"_addr",context))
        var base_id = base.entry.store_id
        var index_id = index.entry.store_id
        with GILReleased(Python()):
            _ = msel_store_take_rows(view.store, base_id, n, row_bytes, index_id, count, dst)
    return output


# REMOVED (lane vdf-scope-cv, 2026-11-24): `arena_resident_entry` (`_Resident._entry`), no caller left.


def arena_materialize(rows: ArenaRows, context: PythonObject) raises -> PythonObject:
    # DeviceRows.materialize: the fold's host rows, gathered at most once by
    # its resident store owner and kept by the fold's native state.
    if rows.host is not None:
        var cached = rows.host
        return cached
    var view = rows.resident_view()
    var output = arena_resident_take(view, rows.base, rows.indices, context)
    var kept = arena_rows_publish_host(rows.word, output, Int(py=_slot(output,"_addr",context)))
    return kept


def arena_rows_store(rows: ArenaRows, context: PythonObject) raises -> DoorRowsStore:
    """The resident store ids of a fold view: the entries of its base and of
    its int64 indices in model selection's store (each put ONCE per owner and
    kept for the fold set), so a device kernel entry gathers the fold out of
    the resident base with no host word read. `context` is the rows context
    (`_arena_io._ROWS_CONTEXT`). No download and no host gather; one lease
    of the resident state for the operation."""
    var view = rows.resident_view()
    var lease = ARENA_RESIDENTS.acquire(arena_handle(view.word))
    var base = _entry_info(rows.base, lease, view, False, context)
    var index = _entry_info(rows.indices, lease, view, True, context)
    var n = base.entry.rows
    var row_bytes = 0
    if n:
        row_bytes = base.entry.nbytes // n
    if row_bytes % 4 != 0:
        raise Error("TypeError: DeviceRows requires a float32 base")
    return DoorRowsStore(base.entry.store_id, n, row_bytes // 4, index.entry.store_id, Int(py=index.info["size"]))
