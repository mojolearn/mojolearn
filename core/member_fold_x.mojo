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
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_I32, MEMBER_I64, MEMBER_SCOPE_NONE, MEMBER_HOME_HOST,
)
from core.entry_table import EntryRef, entry_ref_unresolved, entry_ref_mode, entry_ref_vendor
from core.prep_entry_table import PrepEntryKey, xp_transport_ref
from core.host_parallel import host_widen_index_i64, host_parallelize
from core.fold_x_word import fold_x_word
from bindings.arena_cache_run import (
    ARENA_PREFIX_X_PREP, ARENA_STORE_PUT, arena_scope_store_hold, arena_cache_unhold, arena_store_take_rows,
    arena_cache_free,
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
    var base_rows: Int

    def __init__(out self):
        self.store = entry_ref_unresolved()
        self.slot = -1
        self.rows = 0
        self.cols = 0
        self.index = 0
        self.wide = List[Int64]()
        self.base_rows = 0

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
    fold.base_rows = call[].x.rows
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
    var failure = String("")
    if call[].scope != MEMBER_SCOPE_NONE and call[].pin != 0:
        # HELD (lane gd-device-store): the cached whole X cannot be freed by a
        # concurrent release, drop or close until the gather out of it is done.
        var held = arena_scope_store_hold(call[].scope, fold.store, ARENA_PREFIX_X_PREP, call[].x.data, words,
                                          call[].pin)
        if held.pinned:
            call[].pin_kept = call[].pin  # THE PIN RULE: the cache holds the reference now
        try:
            fold.slot = arena_store_take_rows(fold.store, held.store_id, fold.cols, fold.index, fold.rows)
        except error:
            failure = String(error)
        try:
            arena_cache_unhold(held)  # take_rows waited for the store's context
        except unhold_error:
            if failure.byte_length() == 0:
                failure = String(unhold_error)
    else:
        var source: List[Int] = [call[].x.data]
        var counts: List[Int] = [words]
        var own = xp_transport_call(fold.store, ARENA_STORE_PUT, source, counts)
        try:
            fold.slot = arena_store_take_rows(fold.store, own, fold.cols, fold.index, fold.rows)
        except error:
            failure = String(error)
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
    """Rows `index[0 .. rows)` (int64) of the dense host 4-byte matrix at
    `src` (`width` words a row) into `dst`, bit for bit, in row chunks
    across the host's cores (the targets and weights of a fold; X itself
    only on a sub-route whose entry reads no device X, which the member's
    report names)."""
    member_fold_take_words(src, index, rows, width, dst, 4)


def member_fold_take_view(view: MemberMatrix, index: Int, rows: Int, dst: Int):
    """`member_fold_take_words` of a dense host view (4- or 8-byte words)."""
    member_fold_take_words(view.data, index, rows, view.cols, dst, view.element_bytes())


def member_fold_take_words(src: Int, index: Int, rows: Int, width: Int, dst: Int, bytes: Int):
    """The take of `member_fold_take_f32` for words of `bytes` (4 or 8)
    bytes, copied as integers (no float load, no bit can move)."""
    if rows <= 0 or width <= 0:
        return
    var tasks = (rows + _TAKE_CHUNK - 1) // _TAKE_CHUNK
    var wide = bytes == 8

    def _task(t: Int) {imm src, imm index, imm rows, imm width, imm dst, imm wide}:
        var i = _I64P(unsafe_from_address=index)
        var lo = t * _TAKE_CHUNK
        var hi = min(rows, lo + _TAKE_CHUNK)
        if wide:
            var s = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=src)
            var o = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=dst)
            for r in range(lo, hi):  # the task's rows (one chunk of the split)
                var at = Int(i[r]) * width
                for c in range(width):  # small-loop(one row's words)
                    o[r * width + c] = s[at + c]
        else:
            var s = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=src)
            var o = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=dst)
            for r in range(lo, hi):  # the task's rows (one chunk of the split)
                var at = Int(i[r]) * width
                for c in range(width):  # small-loop(one row's words)
                    o[r * width + c] = s[at + c]

    host_parallelize(_task, tasks)


def member_fold_take_into(view: MemberMatrix, fold: MemberFoldX, mut store: List[UInt64]) -> MemberMatrix:
    """The fold's rows of a dense host column view (targets, class codes of
    any 4- or 8-byte dtype, weights) taken bitwise through `fold.index` into
    `store` (which the caller keeps alive past the operation), as a view of
    the same dtype (lane vdf-scope-linear2, an addition)."""
    var words = fold.rows * view.cols
    store = List[UInt64](length=max((words * view.element_bytes() + 7) // 8, 1), fill=UInt64(0))
    member_fold_take_view(view, fold.index, fold.rows, Int(store.unsafe_ptr()))
    var part = view
    part.data = Int(store.unsafe_ptr())
    part.rows = fold.rows
    return part


struct MemberFoldColumn(Movable):
    """THE PER-INPUT PINS (lane vdf-scope-pins; core/member_abi.mojo
    `MemberCall.y_pin`): one fold's rows of a base column (the targets or the
    sample weights) taken ON THE DEVICE into a slot of the fold's store, as
    `member_fold_x_open` takes X. `slot` -1: none; `row_words` the 4-byte
    words of one row (a float32 or int32 column: its width; float64 / int64:
    twice it)."""
    var slot: Int
    var rows: Int
    var row_words: Int

    def __init__(out self):
        self.slot = -1
        self.rows = 0
        self.row_words = 0

    def word(self) -> Int:
        """The address word of the slot (THE FOLD X WORD numbering,
        core/fold_x_word.mojo): an entry whose upload goes through
        `fold_x_fill` reads it as it reads a fold X word."""
        return fold_x_word(self.slot)


def member_fold_column_open(call: _Call, fold: MemberFoldX, view: MemberMatrix, pin: Int, mut kept: Int,
                            mut column: MemberFoldColumn, what: String, estimator: String) raises:
    """The fold's rows of the dense host column `view` (the WHOLE base: one
    row per row of X, read through `fold.index`) taken on the device into
    `column.slot`. Inside the caller's scope with a nonzero `pin` (the
    column's own token, `call.y_pin` or `call.weights_pin`) the whole column
    is found in the scope's x_prep cache or put ONCE and published with that
    token (`kept` = 1 the moment the cache took it, THE PIN RULE); without a
    pin it is put into a slot this call owns and frees. Only the index words
    and, once per scope, the column cross the bus. Same bytes as
    `member_fold_take_view` (a device gather of the same rows in the same
    order)."""
    # The WHOLE base's rows (`fold.base_rows`, read in `member_fold_index`):
    # the caller has already put the fold's X view into `call.x` here.
    if view.rows != fold.base_rows or view.cols < 1 or view.stride != view.cols or view.offset != 0:
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host column of X's rows")
    if view.home != MEMBER_HOME_HOST:
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host column of X's rows")
    column.rows = fold.rows
    column.row_words = view.cols * view.element_bytes() // 4
    var words = view.rows * column.row_words
    var failure = String("")
    if call[].scope != MEMBER_SCOPE_NONE and pin != 0:
        # HELD (lane gd-device-store), as `member_fold_x_open`.
        var held = arena_scope_store_hold(call[].scope, fold.store, ARENA_PREFIX_X_PREP, view.data, words, pin)
        if held.pinned:
            kept = 1  # THE PIN RULE: the cache holds the reference now
        try:
            column.slot = arena_store_take_rows(fold.store, held.store_id, column.row_words, fold.index, fold.rows)
        except error:
            failure = String(error)
        try:
            arena_cache_unhold(held)  # take_rows waited for the store's context
        except unhold_error:
            if failure.byte_length() == 0:
                failure = String(unhold_error)
    else:
        var source: List[Int] = [view.data]
        var counts: List[Int] = [words]
        var own = xp_transport_call(fold.store, ARENA_STORE_PUT, source, counts)
        try:
            column.slot = arena_store_take_rows(fold.store, own, column.row_words, fold.index, fold.rows)
        except error:
            failure = String(error)
        arena_cache_free(fold.store, own)  # take_rows waited for the store's context
    if failure.byte_length() > 0:
        raise Error(failure)


def member_fold_column_close(fold: MemberFoldX, mut column: MemberFoldColumn) raises:
    """Free the column's fold slot (after the entry that read it returned)."""
    if column.slot >= 0:
        var slot = column.slot
        column.slot = -1
        arena_cache_free(fold.store, slot)

