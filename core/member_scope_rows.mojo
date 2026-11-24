# SPDX-License-Identifier: Apache-2.0
"""The rows of one member call under a host row index, for a family whose image has its own resident store
(lane vdf-scope-cluster2b, 2026-11-24: mixture, density, spectral, manifold).

A model-selection search hands a member that answers `MEMBER_CAN_ROW_INDEX` the WHOLE host X plus the side's row
numbers (core/msel_search_run.mojo, THE RESIDENT ROUTE), and the scope and X's pin to the one image that answers
`MEMBER_CAN_SCOPE`. `member_scope_rows` turns that into the X the family's entries read:

  * RESIDENT (`store_ok`, a scope, `pin != 0`): X found in, or put ONCE into, the scope's cache on the family's
    store (`arena_scope_store_put`, the transport entry in `route.second`; `call.pin_kept` = 1 the moment the
    cache keeps the caller's pin: THE PIN RULE), and the side's rows taken ON THE DEVICE into a fresh slot
    (`arena_store_take_rows`: the int32 rows widened to int64 on the host cores, only those words cross). Every
    fold and candidate of one search reads ONE upload of X.
  * GATHERED otherwise (no scope, no pin, a host image): the rows gathered on the host cores
    (`admitted_gather_rows_f32`), the block the caller would have gathered itself.

Both hold the same rows in the same order (a byte copy either way), so the family's entries compute the same
bits from either. `member_scope_rows_done` frees a taken slot (call it in a `finally`).
"""
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_I32, MEMBER_I64, MEMBER_HOME_DEVICE, MEMBER_SCOPE_NONE,
)
from core.admitted_rows_gather import admitted_gather_rows_f32, ARG_OUT_OF_RANGE
from core.host_parallel import host_widen_index_i64
from core.entry_table import EntryRef
from bindings.arena_cache_run import arena_scope_store_put, arena_store_take_rows, arena_cache_free


struct MemberScopeRows(Movable):
    """The X of one call: `source` is a host address (`resident` False: `gathered`'s words, or the caller's whole
    X without an index) or a slot id of the family store (`resident` True), `rows` x `cols` float32."""
    var source: Int
    var rows: Int
    var cols: Int
    var resident: Bool
    var gathered: List[Float32]

    def __init__(out self, source: Int, rows: Int, cols: Int, resident: Bool, var gathered: List[Float32]):
        self.source = source
        self.rows = rows
        self.cols = cols
        self.resident = resident
        self.gathered = gathered^


def member_scope_rows(mut call: MemberCall, prefix: String, store_ok: Bool, est: String) raises -> MemberScopeRows:
    """The call's X: the whole dense host view without a row index; with one, RESIDENT or GATHERED (module
    note). `prefix` names the family's scope cache, `store_ok` says the route carries the family store in
    `route.second`, `est` names the estimator in a refusal."""
    var store = call.route.second  # a copy: `call` is also passed mutably
    return member_scope_rows_in(call, store, prefix, store_ok, est)


def member_scope_rows_in(mut call: MemberCall, store: EntryRef, prefix: String, store_ok: Bool,
                         est: String) raises -> MemberScopeRows:
    """`member_scope_rows` with the family store's transport entry `store` named by the caller (lane
    vdf-scope-cluster4: a member whose route slots are all in use resolves its image's store elsewhere,
    core/metrics_store_ref.mojo). `store` is read only when `store_ok`."""
    var x = call.x
    if not x.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + est + " member reads X as a dense host float32 view")
    var index = call.row_index
    if index.rows == 0:
        return MemberScopeRows(x.data, x.rows, x.cols, False, List[Float32]())
    if index.home == MEMBER_HOME_DEVICE or (index.dtype != MEMBER_I32 and index.dtype != MEMBER_I64):
        raise Error("TypeError: a " + est + " member reads a host int32 or int64 row index")
    var n = index.rows
    if store_ok and call.scope != MEMBER_SCOPE_NONE and call.pin != 0:
        var put = arena_scope_store_put(call.scope, store, prefix, x.data, x.rows * x.cols, call.pin)
        if put.pinned:
            call.pin_kept = 1
        var wide = List[Int64]()
        var index_address = index.data
        if index.dtype == MEMBER_I32:
            wide = List[Int64](unsafe_uninit_length=max(n, 1))
            host_widen_index_i64(index.data, Int(wide.unsafe_ptr()), n)
            index_address = Int(wide.unsafe_ptr())
        # A row outside X is refused by the store's gather (its status word), in its words.
        var taken = arena_store_take_rows(store, put.store_id, x.cols, index_address, n)
        _ = wide^  # the gather waited for its index copy: the host words may go
        return MemberScopeRows(taken, n, x.cols, True, List[Float32]())
    var dst = List[Float32](unsafe_uninit_length=max(n * x.cols, 1))
    var status: Int
    if index.dtype == MEMBER_I64:
        status = admitted_gather_rows_f32[DType.int64](x.data, x.rows, x.cols, index.data, n, Int(dst.unsafe_ptr()))
    else:
        status = admitted_gather_rows_f32[DType.int32](x.data, x.rows, x.cols, index.data, n, Int(dst.unsafe_ptr()))
    if status == ARG_OUT_OF_RANGE:
        raise Error("IndexError: a " + est + " member's row index holds a row outside X's " + String(x.rows)
            + " rows")
    var rows = MemberScopeRows(0, n, x.cols, False, dst^)
    rows.source = Int(rows.gathered.unsafe_ptr())
    return rows^


def member_scope_rows_done(call: MemberCall, rows: MemberScopeRows) raises:
    """Free a taken slot (RESIDENT); nothing for host rows. The cached whole X stays the scope's."""
    member_scope_rows_done_in(call.route.second, rows)


def member_scope_rows_done_in(store: EntryRef, rows: MemberScopeRows) raises:
    """`member_scope_rows_done` into the store `member_scope_rows_in` took the slot from."""
    if rows.resident:
        arena_cache_free(store, rows.source)
