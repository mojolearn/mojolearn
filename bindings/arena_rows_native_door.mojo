# SPDX-License-Identifier: Apache-2.0
# Native ownership of DeviceRows metadata, without invoking buffer protocols.
#
# Lane arena (2026-11-21): a DeviceRows is one native `DeviceRowsState`
# (`core/arena_state.mojo`, `ARENA_ROWS`) behind the one owner object in the
# instance's `_owner` slot (`bindings/arena_owner_door.mojo`). Every reader in
# every binding resolves it with `arena_rows`; the instance dictionary, its
# class admission and the "store_ids" operation (a door's call back through
# Python) are gone. The public attributes are read-only reads of the state.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.py_balanced import py_call, py_type
from bindings.native_error_class import error_without_class
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.arena_array_metadata_pack import arena_array_metadata
from bindings.arena_resident_rows_door import arena_materialize
from bindings.arena_owner_door import (
    ArenaRows, ARENA_OWNER_ROWS, arena_owner_new, arena_owner_slot, arena_pin, arena_unpin, arena_resident_view,
    arena_rows, arena_pinned,
)
from bindings.native_owner_lifetime_door import native_release_admit, native_release_weakrefs
from bindings.arena_cache_stats_pack import arena_cache_stats
from bindings.arena_cache_get_door import arena_cache_get
from bindings.arena_cache_upload_door import arena_cache_id_of
from bindings.arena_cache_lifecycle_pack import arena_cache_lifecycle
from bindings.arena_cache_binding_pack import arena_cache_supports
from bindings.arena_cache_release_door import arena_cache_release_operation
from bindings.arena_resident_native_door import arena_resident_operation
from bindings.optimizer_public_words_unpack import optimizer_python_error, PYTHON_ERROR_PENDING
from core.arena_state import (
    ARENA_ROWS, ARENA_HANDLE_NONE, ARENA_ROWS_PIN_BASE, ARENA_ROWS_PIN_INDICES, ARENA_ROWS_PIN_RESIDENT,
    DeviceRowsState, arena_handle_word, arena_handle,
)
from core.entry_table import entry_ref_unresolved


def _rows_view(model: PythonObject, context: PythonObject) raises -> ArenaRows:
    # UNPACK of every DeviceRows operation: the fold view, resolved once.
    return arena_rows(model, context["maxabs_rows_owner"], context["maxabs_rows_class"])


def _rows_release_admit(model: PythonObject, base: PythonObject, indices: PythonObject,
                        resident_owner: PythonObject, context: PythonObject) raises:
    """Re-initializing an existing view releases its old pins when the old
    state goes, and a release can run Python (an application finalizer or a
    weak-reference callback). Admit every owner the replacement drops BEFORE
    anything changes, as the instance-dictionary form did (lane fx-leftovers
    item 7, 2026-11-21; unit C had dropped it, so Python could run inside
    `__init__`): the old base, indices and materialized host rows through
    `native_release_admit` unless the new view keeps the same object, and an
    old resident store owner other than the new one through its weak
    references (the old form admitted the bound `take` method the same way).
    A refusal leaves the view unchanged. Once per re-initialization."""
    var old = _rows_view(model, context)
    var old_resident = PythonObject(None)
    var resident_pin = 0
    if True:
        var lease = ARENA_ROWS.acquire(arena_handle(old.word))
        resident_pin = lease[].pins[ARENA_ROWS_PIN_RESIDENT]
    if resident_pin != 0:
        old_resident = arena_pinned(resident_pin)
    if old.base is not base and old.base is not indices:
        native_release_admit(old.base, context)
    if old.indices is not indices and old.indices is not base:
        native_release_admit(old.indices, context)
    if old.host is not None and old.host is not base and old.host is not indices:
        native_release_admit(old.host, context)
    if old_resident is not None and old_resident is not resident_owner:
        native_release_weakrefs(old_resident, context)
    _ = old^


def _rows_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    # UNPACK: the base and index metadata and the resident store owner behind
    # `take`. RUN: the typed state, inserted once. PACK: the one owner object
    # in the instance's `_owner` slot. Runs once per fold view.
    var base = arena_array_metadata(args[0],context)
    var indices = arena_array_metadata(args[1],context)
    var base_shape = base["shape"]
    if len(base_shape) == 0:
        raise Error("IndexError: tuple index out of range")
    var rows = Int(py=indices["size"])
    var size = rows
    var trailing = List[Int]()
    for i in range(1,len(base_shape)):  # driver-loop: shape of the selected rows, preserving all trailing dimensions
        var extent = Int(py=base_shape[i])
        if size > 0 and extent > 9223372036854775807//size:
            raise Error("ValueError: DeviceRows shape exceeds native indexing bounds")
        size *= extent
        trailing.append(extent)
    var first = Int(py=base_shape[0])
    var row_words = 0
    if first != 0:
        row_words = Int(py=base["size"])//first
    var dtype = String()
    var named = base["dtype"]
    if py_type(named) is context["arena_native_types"]["str"]:
        dtype = String(py=named)
    # The gather owner: only the bound `model_selection._Resident.take` is
    # admitted (its native store entry is copied here, once); any other
    # `take` is never invoked and `materialize` refuses it in words.
    var take = args[2]
    var resident = ARENA_HANDLE_NONE
    var store = entry_ref_unresolved()
    var resident_owner = PythonObject(None)
    if py_type(take) is context["owner_method_class"] and take.__func__ is context["arena_resident_take"]:
        resident_owner = take.__self__
        var gather = arena_resident_view(resident_owner, context)
        resident = gather.word
        store = gather.store
    var previous = arena_owner_slot(context["maxabs_rows_owner"], model, context["maxabs_rows_class"],
                                    context["cache_attribute_error"])
    if previous is not None:
        _rows_release_admit(model, args[0], args[1], resident_owner, context)
    var state = DeviceRowsState(0, first, row_words, Int(py=base["itemsize"]), dtype, trailing^, 0, rows, resident)
    state.pins[ARENA_ROWS_PIN_BASE] = arena_pin(args[0])
    state.pins[ARENA_ROWS_PIN_INDICES] = arena_pin(args[1])
    if resident_owner is not None:
        state.pins[ARENA_ROWS_PIN_RESIDENT] = arena_pin(resident_owner)
    var pins = state.pins.copy()
    var word = ARENA_HANDLE_NONE
    try:
        word = arena_handle_word(ARENA_ROWS.insert(state^))
    except error:
        for i in range(len(pins)):  # driver-loop: over the pins of a state that was never inserted
            arena_unpin(pins[i])
        raise error^
    # From here the owner object releases the state when CPython frees it.
    var owner = arena_owner_new(ARENA_OWNER_ROWS, word, store)
    _ = py_call(context["maxabs_rows_owner_set"], model, owner)
    # A reinitialized instance frees its previous view here.
    _ = previous^


def _rows_attribute(model: PythonObject, name: PythonObject, context: PythonObject) raises -> PythonObject:
    # PACK of the read-only public attributes: reads of the native state.
    var b = context["arena_native_types"]
    if py_type(name) is not b["str"]:
        raise Error("TypeError: DeviceRows attribute names are native strings")
    var which = String(py=name)
    var view = _rows_view(model, context)
    if which == "base":
        return view.base
    if which == "indices":
        return view.indices
    if which == "shape":
        var shape = Python.list(PythonObject(view.rows))
        for i in range(len(view.trailing)):  # driver-loop: over the base's trailing dimensions
            _ = py_call(shape.append, PythonObject(view.trailing[i]))
        var extents = py_call(b["tuple"], shape)
        return extents
    if which == "dtype":
        # The base's own dtype object (a string for an Array).
        var named = arena_array_metadata(view.base, context)["dtype"]
        return named
    if which == "ndim":
        return PythonObject(1 + len(view.trailing))
    if which == "size":
        # rows x the trailing extents: the size of `shape`, also when the base
        # has no rows (row_words is 0 there; arena unit C 9b369cd34f answered
        # rows * row_words, 0 for a (0, 3) base and two indices).
        var words = view.rows
        for i in range(len(view.trailing)):  # driver-loop: over the base's trailing dimensions
            words *= view.trailing[i]
        return PythonObject(words)
    if which == "itemsize":
        return PythonObject(view.itemsize)
    if which == "row_words":
        return PythonObject(view.row_words)
    raise Error("AttributeError: 'DeviceRows' object has no attribute '" + which + "'")


def arena_rows_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        var result = PythonObject(None)
        var cache_release_op = (operation == "cache_init" or operation == "cache_release" or operation == "cache_close"
            or operation == "cache_exit" or operation == "scope_init" or operation == "scope_enter" or operation == "scope_exit" or operation == "scope_close")
        if operation == "cache_len" or operation == "resident_stats" or operation == "cache_attribute":
            result = arena_cache_stats(model,operation,args,context)
        elif operation == "cache_get":
            result = arena_cache_get(model,args,context)
        elif operation == "cache_id_of":
            result = arena_cache_id_of(model,args,context)
        elif operation == "cache_supports":
            result = arena_cache_supports(args,context)
        elif operation == "cache_enter" or operation == "cache_live":
            result = arena_cache_lifecycle(model,operation,context)
        elif cache_release_op:
            result = arena_cache_release_operation(model,operation,args,context)
        else:
            classical_connect_numpy(context)
        if operation.startswith("msel_resident_"):
            result = arena_resident_operation(model,operation,args,context)
        elif operation == "init":
            _rows_initialize(model,args,context)
        elif operation == "materialize":
            var fold = _rows_view(model,context)
            result = arena_materialize(fold,context)
        elif operation == "attribute":
            result = _rows_attribute(model,args[0],context)
        elif operation == "len":
            var counted = _rows_view(model,context)
            result = PythonObject(counted.rows)
        elif not cache_release_op and operation != "cache_len" and operation != "resident_stats" and operation != "cache_attribute" and operation != "cache_get" and operation != "cache_id_of" and operation != "cache_enter" and operation != "cache_live" and operation != "cache_supports":
            raise Error("ValueError: unknown native DeviceRows metadata operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING or String(error).startswith("OverflowError:") or String(error).startswith("RuntimeError:"):
            return optimizer_python_error(error)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error,"TypeError"),ExceptionType.TypeError)
        if String(error).startswith("IndexError:"):
            return raise_python_exception(error_without_class(error,"IndexError"),ExceptionType("PyExc_IndexError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error,"AttributeError"),ExceptionType("PyExc_AttributeError"))
        return raise_python_exception(error_without_class(error,"ValueError"),ExceptionType.ValueError)
