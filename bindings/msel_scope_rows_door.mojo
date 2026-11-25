# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Model selection's rows out of the `resident()` scope (lane gd-rf-msel,
2026-11-25): train_test_split's inputs kept ONCE per scope and gathered in
place.

Before, train_test_split opened a `_Resident` per call (msel_resident_new /
_close, bindings/msel_common_door.mojo), put every input into it and freed it
at the end: inside a scope two calls on the same arrays uploaded twice and
`resident.stats()` showed nothing. Now, inside a scope, on a GPU column:

  * the buffer kept is the CALLER'S (`metric_resident_owner`: the admitted
    array itself when it reads the caller's buffer, the caller's array when
    the admission made a word-for-word copy, else nothing is kept and the
    call takes the per-call route below);
  * it is put into the scope's x_prep cache of the column's image
    (`arena_scope_store_put`, prefix "x_prep", the store every GPU image
    ships and the one member_fold_x and the metric doors key the same X in)
    once, pinned by the cache: a hit on every later call of the scope;
  * each side's rows are taken ON THE DEVICE out of that slot
    (`arena_store_take_rows`: only the int64 row numbers go up) into a fresh
    slot, read once into the Array the caller receives (`arena_store_read`)
    and freed.

StratifiedKFold's label codes (`msel_scope_codes_slot`, lane gd-rf-msel door
2): on the program route the n int32 codes went up into the program store once
per split call (bindings/msel_split_plan_run.mojo `MselProgSlot`). Inside a
scope they are kept in the scope's codes cache under the key of the CALLER'S y
buffer (core/msel_scope_codes.mojo), so a second split of the same y finds them
and the plan reads that slot in place.

Outside a scope (one atomic load, `arena_scopes_live`), on the host column,
for a list or tuple, an input of rows that are not whole 4-byte words, or one
below the scope's size floor, nothing here runs and the per-call `_Resident`
route is unchanged. Python objects only at UNPACK (the scope word, the
admitted metadata, the output allocation) and PACK; the store calls run with
the interpreter lock released.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call, py_get, py_type
from bindings.arena_cache_run import (
    ARENA_PREFIX_X_PREP, ArenaCachePut, arena_scopes_live, arena_scope_floor, arena_scope_store_put,
    arena_store_take_rows, arena_store_read, arena_cache_free,
)
from bindings.arena_owner_door import arena_pin, arena_unpin
from bindings.arena_materialize_native_door import _prepared_info, _empty, _empty_unfilled, _slot
from bindings.member_scope_door import member_scope_word
from bindings.metric_resident_source_door import metric_resident_owner, _exact_kind, _c_order
from bindings.msel_split_plan_run import (
    MselSplitRoute, MselCodes, msel_prog_store, msel_stratified_on_slot, msel_default_folds_stored,
)
from core.msel_scope_codes import msel_scope_codes_put, msel_helper_store_ops, MSEL_HELPER_CODES_PREFIX
from bindings.classical_arrays_door import classical_address
from core.arena_state import ARENA_HANDLE_NONE
from core.entry_table import EntryRef, entry_ref_unresolved
from core.prep_entry_table import PrepEntryKey, xp_transport_ref


struct MselScopeRows(Movable):
    """One public operation's view of the calling thread's scope: its handle
    word (`ARENA_HANDLE_NONE`: none, or a column with no device store), the
    x_prep transport entry of the operation's GPU column and the scope's size
    floor in words."""
    var scope: Int
    var store: EntryRef
    var floor: Int

    def __init__(out self):
        self.scope = ARENA_HANDLE_NONE
        self.store = entry_ref_unresolved()
        self.floor = 0

    def active(self) -> Bool:
        return self.scope != ARENA_HANDLE_NONE


def msel_scope_rows(context: PythonObject) raises -> MselScopeRows:
    """UNPACK, once per public operation: the scope view of `context` (the
    model-selection operation context). Outside every scope: one atomic
    load and nothing else. The column is the model-selection helper key's
    (mode, vendor) (vendor 0: the host column, which keeps no scope here)."""
    var held = MselScopeRows()
    if not arena_scopes_live():
        return held^
    var scope = member_scope_word(context["family_msel_rows_context"])
    if scope == ARENA_HANDLE_NONE:
        return held^
    var key = py_get(context["family_msel_entries"], PythonObject("msel_native_key"))
    if key is None or py_type(key) is py_type(PythonObject("")):
        return held^  # no helper key: the per-call route raises its own words
    var vendor = Int(py=key[1])
    if vendor == 0:
        return held^
    try:
        held.store = xp_transport_ref(PrepEntryKey(Int(py=key[0]), vendor))
    except:
        return held^  # the column's image ships no x_prep store: the per-call route
    held.floor = arena_scope_floor(scope)
    held.scope = scope
    return held^


def _scope_take(rows: MselScopeRows, base_id: Int, row_words: Int, info: PythonObject, indices: PythonObject,
                rc: PythonObject) raises -> PythonObject:
    """Rows `indices` (a prepared 1-D int64 Array) of the resident slot
    `base_id` into a new Array of the base's dtype and trailing shape: taken
    on the device, read down once, the fold slot freed."""
    var meta = _prepared_info(indices, rc)
    if String(meta["dtype"]) != "<i8" or Int(py=meta["ndim"]) != 1:
        raise Error("TypeError: model-selection rows require one-dimensional int64 indices")
    var count = Int(py=meta["size"])
    var shape = Python.list(PythonObject(count))
    var base_shape = info["shape"]
    for i in range(1, len(base_shape)):  # driver-loop: the trailing dimensions of the row shape
        _ = py_call(shape.append, base_shape[i])
    var b = rc["arena_native_types"]
    var dtype = String(info["dtype"])
    if count == 0:
        return _empty(py_call(b["tuple"], shape), dtype, rc)
    # Every byte of the output is written by the read below, or the call
    # raises and the output is dropped: no fill pass.
    var output = _empty_unfilled(py_call(b["tuple"], shape), dtype, rc)
    var dst = Int(py=_slot(output, "_addr", rc))
    var index_address = Int(py=meta["address"])
    var store = rows.store
    var words = count * row_words
    with GILReleased(Python()):
        var slot = arena_store_take_rows(store, base_id, row_words, index_address, count)
        try:
            arena_store_read(store, slot, dst, words)
        finally:
            arena_cache_free(store, slot)  # the read waited for the store's context
    _ = indices  # its int64 words were read through index_address above
    return output


def msel_scope_take_pair(rows: MselScopeRows, source: PythonObject, prepared: PythonObject, train: PythonObject,
                         test: PythonObject, context: PythonObject) raises -> PythonObject:
    """`(rows train, rows test)` of one train_test_split input out of the
    scope (module note), or None when the input is not kept (the caller takes
    the per-call route). `source` is the caller's argument, `prepared` its
    admitted Array. Asked only when `rows.active()`."""
    var rc = context["family_msel_resident_context"]
    if py_type(prepared) is not rc["array_class"]:
        return PythonObject(None)
    var owner = metric_resident_owner(source, prepared)
    if owner is None:
        return PythonObject(None)
    var info = _prepared_info(prepared, rc)
    var shape = info["shape"]
    if len(shape) == 0:
        return PythonObject(None)
    var n = Int(py=shape[0])
    var nbytes = Int(py=info["nbytes"])
    if n <= 0 or nbytes % n != 0:
        return PythonObject(None)
    var row_bytes = nbytes // n
    if row_bytes == 0 or row_bytes % 4 != 0:
        return PythonObject(None)
    var words = nbytes // 4
    if words < rows.floor:
        return PythonObject(None)
    var address = Int(py=classical_address(owner))
    # THE PIN RULE: one reference on the kept buffer's owner; the cache keeps
    # it when it published this put, else it is dropped here.
    var pin = arena_pin(owner)
    var put = ArenaCachePut(-1, False)
    var scope = rows.scope
    var store = rows.store
    try:
        with GILReleased(Python()):
            put = arena_scope_store_put(scope, store, String(ARENA_PREFIX_X_PREP), address, words, pin)
    except error:
        arena_unpin(pin)
        raise error^
    if not put.pinned:
        arena_unpin(pin)
    var row_words = row_bytes // 4
    var first = _scope_take(rows, put.store_id, row_words, info, train, rc)
    var second = _scope_take(rows, put.store_id, row_words, info, test, rc)
    _ = owner  # its words are the slot's source; the cache pins it from here
    _ = prepared
    return Python.tuple(first, second)


def msel_scope_codes_slot(y: PythonObject, route: MselSplitRoute, labels: MselCodes, form: Int, k: Int,
                          context: PythonObject) raises -> Int:
    """StratifiedKFold's codes slot kept by the calling thread's scope
    (module note), or -1 (no scope, the host column, a plan that reads no
    program slot, or a y that is not the caller's own exact C-ordered buffer
    of whole 4-byte words: the plan then puts and frees its own slot as
    before). `y` is the label argument the codes were encoded from;
    `labels` the codes, kept alive by the caller through the plan. THE PIN
    RULE: one reference on y, kept by the cache when it published this put,
    else dropped here."""
    var rows = msel_scope_rows(context)
    if not rows.active() or not msel_stratified_on_slot(route, labels, form, k):
        return -1
    var kind = _exact_kind(y)
    if kind == 0 or not _c_order(y, kind):
        return -1
    if Int(py=y.size) != labels.n:
        return -1
    var nbytes = labels.n * Int(py=y.itemsize)
    if nbytes == 0 or nbytes % 4 != 0 or nbytes // 4 < rows.floor:
        return -1
    var address = Int(py=classical_address(y))
    var store = msel_prog_store(route)
    var pin = arena_pin(y)
    var put = ArenaCachePut(-1, False)
    var scope = rows.scope
    var codes_address = labels.address
    var n = labels.n
    var words = nbytes // 4
    try:
        with GILReleased(Python()):
            put = msel_scope_codes_put(scope, store, address, words, codes_address, n, pin)
    except error:
        arena_unpin(pin)
        raise error^
    if not put.pinned:
        arena_unpin(pin)
    _ = y  # its address is the key above; the cache pins it from here
    return put.store_id


def _caller_key_words(y: PythonObject, n: Int, floor: Int) raises -> Int:
    """The 4-byte word count of the caller's label buffer `y` as a scope key,
    or -1 when it is not the caller's own exact C-ordered buffer of `n`
    items in whole 4-byte words at or above the scope's floor."""
    var kind = _exact_kind(y)
    if kind == 0 or not _c_order(y, kind):
        return -1
    if Int(py=y.size) != n:
        return -1
    var nbytes = n * Int(py=y.itemsize)
    if nbytes == 0 or nbytes % 4 != 0 or nbytes // 4 < floor:
        return -1
    return nbytes // 4


def msel_scope_helper_codes_id(y: PythonObject, route: MselSplitRoute, n: Int, k: Int, codes: MselCodes,
                               coded: Bool, sabotage: Bool, context: PythonObject) raises -> Int:
    """The stratified default folds' codes block kept by the calling
    thread's scope in the model-selection helper table's own store (prefix
    MSEL_HELPER_CODES_PREFIX, its operation numbers registered through the
    per-family op map), keyed on the caller's y; or -1 (no scope, the host
    column, a plan that puts no codes, or a y that is not the caller's own
    buffer: the plan puts and frees its own block as before). THE PIN RULE as
    in `msel_scope_codes_slot`."""
    var rows = msel_scope_rows(context)
    if not rows.active() or not msel_default_folds_stored(n, k, codes, coded, sabotage):
        return -1
    var words = _caller_key_words(y, codes.n, rows.floor)
    if words < 0:
        return -1
    var address = Int(py=classical_address(y))
    var store = route.helpers
    var pin = arena_pin(y)
    var put = ArenaCachePut(-1, False)
    var scope = rows.scope
    var codes_address = codes.address
    var count = codes.n
    try:
        with GILReleased(Python()):
            put = msel_scope_codes_put(scope, store, msel_helper_store_ops(), String(MSEL_HELPER_CODES_PREFIX),
                                       address, words, codes_address, count, pin)
    except error:
        arena_unpin(pin)
        raise error^
    if not put.pinned:
        arena_unpin(pin)
    _ = y  # its address is the key above; the cache pins it from here
    return put.store_id
