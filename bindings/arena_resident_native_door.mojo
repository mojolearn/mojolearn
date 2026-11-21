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
from core.arena_state import (
    ARENA_RESIDENTS, ArenaResidentState, ArenaResidentEntry, ArenaStoreFree, arena_handle, arena_handle_word,
    arena_resident_close_entries, arena_resident_release_entry,
)
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


struct _ResidentStore(ArenaStoreFree):
    # The fold store's free through the entry resolved when the owner was
    # built, and the operation's first failure (kept with its Python error).
    var store: EntryRef
    var first: FirstFailure

    def __init__(out self, view: ArenaOwnerView):
        self.store = view.store
        self.first = FirstFailure()

    def free(mut self, store_id: Int) raises:
        if store_id >= 0:
            # The store's own refusal text is retained unchanged.
            with GILReleased(Python()):
                _ = msel_store_free(self.store, store_id)

    def failed(mut self, error: Error) raises:
        self.first.keep(error)


def _close(view: ArenaOwnerView) raises:
    # The failure order is core/arena_state.mojo `arena_resident_close_entries`
    # (checked natively by checks/native_tests/arena_state_suite.mojo; lane
    # fx-leftovers item 9: it was reached only through the `msel_free` test export).
    var lease = ARENA_RESIDENTS.acquire(arena_handle(view.word))
    var store = _ResidentStore(view)
    var drained = List[ArenaResidentEntry]()
    var kept = arena_resident_close_entries(lease[], store, drained)
    # Owners are released here, after every free and publication.
    for i in range(len(drained)):  # driver-loop: over the drained entries' pins
        if not kept[i]:
            arena_unpin(drained[i].owner_pin)
            arena_unpin(drained[i].prepared_pin)
    store.first.finish()


def _release(view: ArenaOwnerView, values: PythonObject) raises:
    var lease = ARENA_RESIDENTS.acquire(arena_handle(view.word))
    var store = _ResidentStore(view)
    # Unpublish before the GIL-releasing free. Identity is the native
    # address (CPython's id()); the entry's pin keeps it from being reused.
    var removed = ArenaResidentEntry(0, 0, 0, 0, False, -2, 0, 0)
    var kept = arena_resident_release_entry(lease[], store, Int(values._obj_ptr), removed)
    if removed.store_id == -2:
        return
    if not kept:
        arena_unpin(removed.owner_pin)
        arena_unpin(removed.prepared_pin)
    store.first.finish()


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
