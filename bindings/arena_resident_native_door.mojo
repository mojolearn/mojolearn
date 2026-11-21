# SPDX-License-Identifier: Apache-2.0
# Native model_selection._Resident lifecycle: constructor/reinitialization,
# entry, take, release, close and context enter/exit, one native call each.
#
# State (lane arena, 2026-11-21): the entries live in the native resident
# state of `core/arena_state.mojo` (`ARENA_RESIDENTS`), reached through the
# ONE native owner object the Python class holds in its `_owner` slot
# (`bindings/arena_owner_door.mojo`). UNPACK is the slot read and the
# argument admission; RUN is the typed find / publish / remove / drain of
# the state and the fold store's put / take_rows / free through the
# model-selection helper entry resolved once when the owner was built
# (`core/msel_helpers_calls.mojo`); PACK is the returned Array or tuple.
# Before, every operation read the instance dictionary, re-admitted the
# class and its key types, and called the store through three Python
# builtins held in that dictionary.
#
# Lifetime and error order are unchanged: the entries are unlinked before
# the first GIL-releasing free, every slot is attempted, the first failure
# is retained, failed entries are published again for a retry, and the
# owners (NumPy, pandas, Arrow-backed pandas, ...) are released last, after
# every free and publication, so an owner's own finalizer cannot observe a
# half-closed resident or replace the first error.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call, py_type, py_get
from bindings.arena_cache_release_door import FirstFailure
from bindings.arena_owner_door import (
    ArenaOwnerView, ARENA_OWNER_RESIDENT, arena_owner_new, arena_owner_slot, arena_owner_view, arena_resident_view,
    arena_unpin,
)
from bindings.arena_resident_rows_door import arena_resident_take, arena_resident_entry
from bindings.msel_helpers_key_unpack import msel_helpers_key
from core.arena_state import ARENA_RESIDENTS, ArenaResidentState, arena_handle, arena_handle_word
from core.entry_table import EntryRef
from core.msel_helpers_calls import msel_store_free


def _store_entry(context: PythonObject) raises -> EntryRef:
    # The fold store's entry (put / take_rows / free) of the base binding the
    # model-selection bridge recorded at import: resolved here, once per
    # owner, never per operation.
    var inputs = py_get(context, "arena_msel_input_context")
    if inputs is None:
        raise Error("RuntimeError: native resident rows require the model-selection input connection")
    return msel_helpers_key(inputs)


def _free(view: ArenaOwnerView, store_id: Int) raises:
    if store_id >= 0:
        # The store's own refusal text is retained unchanged.
        with GILReleased(Python()):
            _ = msel_store_free(view.store, store_id)


def _close(view: ArenaOwnerView) raises:
    var lease = ARENA_RESIDENTS.acquire(arena_handle(view.word))
    # Unlink everything first: no other thread finds an entry being freed.
    var drained = lease[].drain()
    var first = FirstFailure()
    var kept = List[Bool](length=len(drained), fill=False)
    for i in range(len(drained)):  # driver-loop: attempt every independent release (one small device free each)
        try:
            _free(view, drained[i].store_id)
        except error:
            first.keep(error)
            # A failed entry stays owned for a retry.
            try:
                _ = lease[].publish(drained[i])
                kept[i] = True
            except publish_error:
                first.keep(publish_error)
    # Owners are released here, after every free and publication.
    for i in range(len(drained)):  # driver-loop: over the drained entries' pins
        if not kept[i]:
            arena_unpin(drained[i].owner_pin)
            arena_unpin(drained[i].prepared_pin)
    first.finish()


def _release(view: ArenaOwnerView, values: PythonObject) raises:
    var lease = ARENA_RESIDENTS.acquire(arena_handle(view.word))
    # Unpublish before the GIL-releasing free. Identity is the native
    # address (CPython's id()); the entry's pin keeps it from being reused.
    var removed = lease[].remove(Int(values._obj_ptr))
    if removed.store_id == -2:
        return
    var first = FirstFailure()
    var kept = False
    try:
        _free(view, removed.store_id)
    except error:
        first.keep(error)
        try:
            _ = lease[].publish(removed)
            kept = True
        except publish_error:
            first.keep(publish_error)
    if not kept:
        arena_unpin(removed.owner_pin)
        arena_unpin(removed.prepared_pin)
    first.finish()


def _initialize(model: PythonObject, context: PythonObject) raises:
    var previous = arena_owner_slot(context["arena_resident_owner_get"], model, context["arena_resident_class"],
                                    context["cache_attribute_error"])
    if previous is not None:
        # Reinitialization frees the live entries. A failed free keeps the
        # old owner in the slot with the failed entries published.
        _close(arena_owner_view(previous, ARENA_OWNER_RESIDENT))
    var store = _store_entry(context)
    var word = arena_handle_word(ARENA_RESIDENTS.insert(ArenaResidentState()))
    # From here the owner object releases the state when CPython frees it.
    var owner = arena_owner_new(ARENA_OWNER_RESIDENT, word, store)
    _ = py_call(context["arena_resident_owner_set"], model, owner)
    # The replaced owner (if any) is freed here: its state is empty.
    _ = previous^


def arena_resident_operation(model: PythonObject, operation: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    if operation == "msel_resident_init":
        _initialize(model, context)
        return PythonObject(None)
    var view = arena_resident_view(model, context)
    if operation == "msel_resident_enter":
        return model
    if operation == "msel_resident_take":
        return arena_resident_take(view, args[0], args[1], context)
    if operation == "msel_resident_entry":
        var dtype = args[1]
        var indices = False
        if dtype is not None:
            if py_type(dtype) is not context["arena_native_types"]["str"] or String(py=dtype) != "<i8":
                raise Error("ValueError: resident entries cast only to the int64 index layout")
            indices = True
        return arena_resident_entry(view, args[0], indices, context)
    if operation == "msel_resident_release":
        _release(view, args[0])
        return PythonObject(None)
    if operation == "msel_resident_close":
        _close(view)
        return PythonObject(None)
    if operation == "msel_resident_exit":
        _close(view)
        return PythonObject(False)
    raise Error("ValueError: unknown native resident lifecycle operation")
