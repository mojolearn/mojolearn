# SPDX-License-Identifier: Apache-2.0
"""THE RESIDENT ROUTE of the x_sequence members (lane vdf-scope-neural, 2026-11-24; core/member_abi.mojo
MEMBER_CAN_ROW_INDEX / MEMBER_CAN_SCOPE): where one member operation's X rows are.

A GPU image publishes its member device store (sequence/seq_member_store.mojo, core/xseq_entry_table.mojo
`xseq_store_ref`); the member's UNPACK puts that entry in `route.fifth` and sets `SEQ_ROUTE_STORE` in
`route.flags`. On such a route an operation reads:

  * no row index: the caller's X. With a scope and a pin (THE PIN RULE) and X at the scope's floor or above, X is
    put ONCE in the scope's "x_sequence" cache (bindings/arena_cache_run.mojo `arena_scope_store_put`) and the
    operation reads that slot in place (X word `-(slot + 1)`, resolved by the device dispatch,
    sequence/model_device_dispatch.mojo `seq_x_slot_resolve`).
  * a row index (`call.row_index`, a dense host int32 / int64 vector of rows of the WHOLE X): with a scope and a
    pin, the whole X is found or put resident in the scope's cache and the fold is taken ON THE DEVICE
    (`arena_store_take_rows`, a word gather into a fresh slot this operation frees after its run); without a pin
    (nothing the cache may keep) the fold's rows are gathered on the host, split across its cores, and go up as
    the gathered route's block did. The target (`y`, `y_cols` 4-byte words a row) is then the whole X's and the
    fold's words are taken on the host in the same row tasks (labels: n * y_cols words, the naive Bayes members'
    label take).

Every form reads the same words in the same order as the host-gathered block, so the fit and predict bits are the
gathered route's and the CPU column's. A route without the store (the host image) answers no row index and keeps
X on the host as before.
"""
from std.memory import unsafe_memcpy
from core.device_route import DeviceRoute
from core.xseq_entry_table import XseqEntryKey, xseq_store_ref
from core.member_abi import MemberCall, MemberMatrix, MEMBER_I32, MEMBER_I64
from core.host_parallel import host_parallelize, host_widen_index_i64
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from bindings.arena_cache_run import (
    arena_scope_floor, arena_scope_store_put, arena_store_take_rows, arena_cache_free,
)

#: `route.flags` bit: `route.fifth` is the image's member device store.
comptime SEQ_ROUTE_STORE = 1048576
#: The scope cache's family prefix of the x_sequence members.
comptime SEQ_STORE_PREFIX = "x_sequence"

comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


def seq_rows_route(route: DeviceRoute) -> Bool:
    """The route holds the image's member store: the kind answers
    MEMBER_CAN_ROW_INDEX and MEMBER_CAN_SCOPE."""
    return (route.flags & SEQ_ROUTE_STORE) != 0 and route.has_fifth


struct SeqRows(Movable):
    """One operation's rows: `n` rows of `d` words; `x` the X word the
    descriptor takes (a host address, or `-(slot + 1)` of the store); `y`
    the target's host address (the fold's words under a row index; 0: none);
    `taken` the fold slot this operation frees (-1: none)."""
    var n: Int
    var d: Int
    var x: Int
    var y: Int
    var taken: Int
    var gathered: List[Float32]
    var y_fold: List[Int32]
    var wide: List[Int64]

    def __init__(out self, n: Int, d: Int, x: Int, y: Int):
        self.n = n
        self.d = d
        self.x = x
        self.y = y
        self.taken = -1
        self.gathered = List[Float32]()
        self.y_fold = List[Int32]()
        self.wide = List[Int64]()


def _take_host(source: Int, wide: Bool, n: Int, base: Int, src: Int, row_words: Int, dst: Int) raises:
    """Rows `index` of the `base` rows of `row_words` 4-byte words at `src`
    into `dst`, bounds-checked, split across the host's cores."""
    var tasks = host_predict_task_count_work(n, n * row_words)
    var chunk = host_predict_chunk(n, tasks)
    var bad = List[Int](length=tasks, fill=0)
    var flags = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(bad.unsafe_ptr()))
    var from_words = _I32P(unsafe_from_address=src)
    var into = _I32P(unsafe_from_address=dst)

    def task(worker: Int) {imm chunk, imm n, imm base, imm wide, imm source, imm flags, imm from_words, imm into,
                           imm row_words}:
        var begin = worker * chunk
        var end = min(begin + chunk, n)
        var out_of_range = 0
        for r in range(begin, end):
            var row = Int(_I64P(unsafe_from_address=source)[r]) if wide else Int(_I32P(unsafe_from_address=source)[r])
            if row < 0 or row >= base:
                out_of_range = 1
                continue
            unsafe_memcpy(dest=into + r * row_words, src=from_words + row * row_words, count=row_words)
        flags[worker] = out_of_range

    host_parallelize(task, tasks)
    for w in range(tasks):  # driver-loop: over the workers' range flags
        if bad[w] != 0:
            raise Error("IndexError: a row index is out of range for X's " + String(base) + " rows")


def seq_rows_open(call: _CallP, y_cols: Int, estimator: String) raises -> SeqRows:
    """The rows of one operation (module note). `y_cols`: the target's
    4-byte words a row to take with a row index (0: the operation reads no
    target). Call `seq_rows_close` after the operation's run, raised or not."""
    if call[].column_index.rows > 0:
        raise Error("TypeError: an " + estimator + " member reads every column of X (no column index)")
    var route = call[].route
    var store_route = seq_rows_route(route)
    var base = call[].x.rows
    var d = call[].x.cols
    var index = call[].row_index
    if index.rows == 0:
        var plain = SeqRows(base, d, call[].x.data, call[].y.data if y_cols > 0 else 0)
        if store_route and call[].scope >= 0 and call[].pin != 0 and base * d >= arena_scope_floor(call[].scope):
            var put = arena_scope_store_put(call[].scope, route.fifth, String(SEQ_STORE_PREFIX), call[].x.data,
                                            base * d, call[].pin)
            if put.pinned:
                call[].pin_kept = 1
            plain.x = -(put.store_id + 1)
        return plain^
    if not ((index.dtype == MEMBER_I32 or index.dtype == MEMBER_I64) and index.cols == 1
            and index.dense_host(index.dtype)):
        raise Error("TypeError: an " + estimator + " member reads its row index as one host int32 or int64 column")
    var n = index.rows
    var wide = index.dtype == MEMBER_I64
    var rows = SeqRows(n, d, 0, 0)
    if store_route and call[].scope >= 0 and call[].pin != 0:
        var idx = index.data
        if not wide:
            rows.wide = List[Int64](unsafe_uninit_length=n)
            host_widen_index_i64(index.data, Int(rows.wide.unsafe_ptr()), n)
            idx = Int(rows.wide.unsafe_ptr())
        var put = arena_scope_store_put(call[].scope, route.fifth, String(SEQ_STORE_PREFIX), call[].x.data,
                                        base * d, call[].pin)
        if put.pinned:
            call[].pin_kept = 1
        rows.taken = arena_store_take_rows(route.fifth, put.store_id, d, idx, n)
        rows.x = -(rows.taken + 1)
    else:
        rows.gathered = List[Float32](length=max(n * d, 1), fill=Float32(0))
        _take_host(index.data, wide, n, base, call[].x.data, d, Int(rows.gathered.unsafe_ptr()))
        rows.x = Int(rows.gathered.unsafe_ptr())
    if y_cols > 0 and call[].y.rows > 0:
        if call[].y.rows != base:
            raise Error("ValueError: X and y have different numbers of rows")
        rows.y_fold = List[Int32](length=max(n * y_cols, 1), fill=Int32(0))
        _take_host(index.data, wide, n, base, call[].y.data, y_cols, Int(rows.y_fold.unsafe_ptr()))
        rows.y = Int(rows.y_fold.unsafe_ptr())
    return rows^


def seq_rows_close(call: _CallP, mut rows: SeqRows) raises:
    """Free the fold slot an operation took on the device (its launches are
    done: the store's free synchronizes the shared sequence context)."""
    if rows.taken >= 0:
        var taken = rows.taken
        rows.taken = -1
        arena_cache_free(call[].route.fifth, taken)


def seq_rows_count(call: _CallP) -> Int:
    """The rows an operation reads: the row index's length, else X's."""
    return call[].row_index.rows if call[].row_index.rows > 0 else call[].x.rows


def seq_rows_target(call: _CallP, view: MemberMatrix, cols: Int) raises -> List[Int32]:
    """The fold's words of a per-row view of the whole X (the target or the
    sample weights: `cols` 4-byte words a row) under a row index, taken in
    host row tasks; empty without a row index (the caller reads the view as
    it is)."""
    var index = call[].row_index
    if index.rows == 0:
        return List[Int32]()
    if view.rows != call[].x.rows:
        raise Error("ValueError: X and y have different numbers of rows")
    var out = List[Int32](length=max(index.rows * cols, 1), fill=Int32(0))
    _take_host(index.data, index.dtype == MEMBER_I64, index.rows, view.rows, view.data, cols,
               Int(out.unsafe_ptr()))
    return out^


def seq_rows_admit(mut route: DeviceRoute):
    """A member's UNPACK, once: the route's image's member device store
    (core/xseq_entry_table.mojo `xseq_store_ref`) into `route.fifth` with
    `SEQ_ROUTE_STORE`, on a GPU image that publishes it; any other image
    (the host image publishes none) keeps the route as it was, so its kind
    answers no row index."""
    if route.vendor <= 0:
        return
    try:
        route.fifth = xseq_store_ref(XseqEntryKey(route.mode, route.vendor))
        route.has_fifth = True
        route.flags |= SEQ_ROUTE_STORE
    except:
        pass
