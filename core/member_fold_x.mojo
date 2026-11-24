# SPDX-License-Identifier: Apache-2.0
"""A member's fold X on the device (lane vdf-scope-linear, 2026-11-24): the
member side of THE FOLD X WORD (core/fold_x_device.mojo).

A member kind that answers `MEMBER_CAN_ROW_INDEX` and `MEMBER_CAN_SCOPE`
(core/member_abi.mojo) is handed the WHOLE dense host float32 X, the fold's
row numbers (`call.row_index`, host int32 or int64) and, inside a
`resident()` scope, the scope word and X's pin. `member_fold_x_open`:

  1. the x_prep family store of the route's image (`xp_transport_ref`, the
     store every GPU image ships and the one the scope's x_prep cache holds,
     so a naive Bayes or discriminant member over the same X shares the
     upload): X is found in the scope's cache, or put ONCE and published
     with the pin (bindings/arena_cache_run.mojo `arena_scope_store_put`,
     lane vdf-scope-run's API; `.pinned` -> `call.pin_kept`, THE PIN RULE).
     Outside a scope, or with no pin, X is put into a slot this call owns
     and frees (what the family's own door does outside a scope);
  2. the fold's rows taken ON THE DEVICE into a fresh slot
     (`arena_store_take_rows`; only the index words cross the bus);
  3. `word()`: the negative X address word the family's entry reads.

`member_fold_x_close` frees the fold slot after the entry returned (the
entry synchronized its own stream). The targets and sample weights are
COLUMNS OF THE BASE read through the same index (`member_fold_take_f32`:
rows x width words, in chunks across the host's cores; the y words cross
the bus as the entry uploads them, never X).
"""
from core.member_abi import MemberCall, MemberMatrix, MEMBER_F32, MEMBER_I32, MEMBER_I64, MEMBER_SCOPE_NONE
from core.entry_table import EntryRef, entry_ref_unresolved, entry_ref_mode, entry_ref_vendor
from core.prep_entry_table import PrepEntryKey, xp_transport_ref
from core.host_parallel import host_widen_index_i64, host_parallelize
from core.fold_x_device import fold_x_word
from bindings.arena_cache_run import (
    ARENA_PREFIX_X_PREP, ARENA_STORE_PUT, arena_scope_store_put, arena_store_take_rows, arena_cache_free,
)
from core.prep_entry_table import xp_transport_call

comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
#: Rows of one host take task (a cache-sized block of a column).
comptime _TAKE_CHUNK = 16384


struct MemberFoldX(Movable):
    """One member operation's fold X: the store entry, the fold slot (-1:
    none), the fold's shape and its int64 row numbers (the caller's, or the
    widened int32 ones this value owns)."""
    var store: EntryRef
    var slot: Int
    var rows: Int
    var cols: Int
    var index: Int
    var wide: List[Int64]

    def __init__(out self):
        self.store = entry_ref_unresolved()
        self.slot = -1
        self.rows = 0
        self.cols = 0
        self.index = 0
        self.wide = List[Int64]()

    def word(self) -> Int:
        """The X address word of the fold slot (core/fold_x_device.mojo)."""
        return fold_x_word(self.slot)


def member_fold_store(call: _Call) raises -> EntryRef:
    """The x_prep family store of the image the member's route resolved to
    (same mode and vendor words); raises when that image ships none."""
    var route = call[].route
    return xp_transport_ref(PrepEntryKey(entry_ref_mode(route.entry), entry_ref_vendor(route.entry)))


def member_fold_store_ready(call: _Call) -> Bool:
    """Whether the route's image ships the x_prep family store (one table
    lookup; asked with the kind's capabilities, once per member class)."""
    try:
        _ = member_fold_store(call)
        return True
    except:
        return False


def member_fold_index(call: _Call, mut fold: MemberFoldX, estimator: String) raises:
    """The fold's int64 row numbers (an int32 index widened once across the
    host's cores) and its shape."""
    var index = call[].row_index
    if index.rows <= 0 or index.cols != 1:
        raise Error("TypeError: a " + estimator + " member reads its row index as one host int32 or int64 column")
    if index.dense_host(MEMBER_I32):
        fold.wide = List[Int64](unsafe_uninit_length=index.rows)
        host_widen_index_i64(index.data, Int(fold.wide.unsafe_ptr()), index.rows)
        fold.index = Int(fold.wide.unsafe_ptr())
    elif index.dense_host(MEMBER_I64):
        fold.index = index.data
    else:
        raise Error("TypeError: a " + estimator + " member reads its row index as one host int32 or int64 column")
    if not call[].x.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads X as a dense host float32 view")
    fold.rows = index.rows
    fold.cols = call[].x.cols
    _index_check(fold.index, fold.rows, call[].x.rows, estimator)


def _index_check(index: Int, rows: Int, base_rows: Int, estimator: String) raises:
    """Every row number inside the base (the host takes of y and the
    weights read through it), in chunks across the host's cores."""
    var tasks = (rows + _TAKE_CHUNK - 1) // _TAKE_CHUNK
    var bad = List[UInt8](length=max(tasks, 1), fill=UInt8(0))
    var bad_at = Int(bad.unsafe_ptr())

    def _task(t: Int) {imm index, imm rows, imm base_rows, imm bad_at}:
        var i = _I64P(unsafe_from_address=index)
        var lo = t * _TAKE_CHUNK
        var hi = min(rows, lo + _TAKE_CHUNK)
        for r in range(lo, hi):  # the task's rows (one chunk of the split)
            var v = Int(i[r])
            if v < 0 or v >= base_rows:
                MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=bad_at)[t] = UInt8(1)
                return

    host_parallelize(_task, tasks)
    for t in range(tasks):  # small-loop(one flag per task)
        if bad[t] != UInt8(0):
            raise Error("IndexError: a " + estimator + " member's row index is outside X")


def member_fold_x_open(call: _Call, mut fold: MemberFoldX, estimator: String) raises:
    """Steps 1-2 of the module note: X resident (the scope's cache, or this
    call's own slot), the fold's rows taken on the device into `fold.slot`."""
    member_fold_index(call, fold, estimator)
    fold.store = member_fold_store(call)
    var words = call[].x.rows * fold.cols
    var base = -1
    var own = -1
    if call[].scope != MEMBER_SCOPE_NONE and call[].pin != 0:
        var put = arena_scope_store_put(call[].scope, fold.store, ARENA_PREFIX_X_PREP, call[].x.data, words,
                                        call[].pin)
        if put.pinned:
            call[].pin_kept = call[].pin  # THE PIN RULE: the cache holds the reference now
        base = put.store_id
    else:
        var source: List[Int] = [call[].x.data]
        var counts: List[Int] = [words]
        own = xp_transport_call(fold.store, ARENA_STORE_PUT, source, counts)
        base = own
    var failure = String("")
    try:
        fold.slot = arena_store_take_rows(fold.store, base, fold.cols, fold.index, fold.rows)
    except error:
        failure = String(error)
    if own >= 0:
        arena_cache_free(fold.store, own)  # take_rows waited for the store's context
    if failure.byte_length() > 0:
        raise Error(failure)


def member_fold_x_close(mut fold: MemberFoldX) raises:
    """Free the fold slot (after the entry that read it returned)."""
    if fold.slot >= 0:
        var slot = fold.slot
        fold.slot = -1
        arena_cache_free(fold.store, slot)


def member_fold_take_f32(src: Int, index: Int, rows: Int, width: Int, dst: Int):
    """Rows `index[0 .. rows)` (int64) of the dense host float32 matrix at
    `src` (`width` words a row) into `dst`, in row chunks across the host's
    cores (the targets and weights of a fold; X itself only on a sub-route
    whose entry reads no device X, which the member's report names)."""
    if rows <= 0 or width <= 0:
        return
    var tasks = (rows + _TAKE_CHUNK - 1) // _TAKE_CHUNK

    def _task(t: Int) {imm src, imm index, imm rows, imm width, imm dst}:
        var s = _F32P(unsafe_from_address=src)
        var i = _I64P(unsafe_from_address=index)
        var o = _F32P(unsafe_from_address=dst)
        var lo = t * _TAKE_CHUNK
        var hi = min(rows, lo + _TAKE_CHUNK)
        for r in range(lo, hi):  # the task's rows (one chunk of the split)
            var at = Int(i[r]) * width
            for c in range(width):  # small-loop(one row's words)
                o[r * width + c] = s[at + c]

    host_parallelize(_task, tasks)
