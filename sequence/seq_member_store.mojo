# SPDX-License-Identifier: Apache-2.0
"""The x_sequence GPU image's member device store (lane vdf-scope-neural, 2026-11-24): the family store behind the
neural members' THE RESIDENT ROUTE (core/member_abi.mojo MEMBER_CAN_ROW_INDEX / MEMBER_CAN_SCOPE).

A cross-validation or a search over MLP members keeps X on the device ONCE in the caller's `resident()` scope
(bindings/arena_cache_run.mojo `arena_scope_store_put`, the scope's cache of the family "x_sequence"), takes each
fold's rows on the device (`take_rows`, core/device_store.mojo: a word gather, the rows never visit the host), and
the member's fit / predict reads that slot in place: the device dispatch (sequence/model_device_dispatch.mojo
`seq_x_slot_resolve`) turns an X address word `-(slot + 1)` into the slot's device pointer.

The store lives on the image's ONE shared sequence context (`sequence_ctx`), the context every `DeviceExec` of the
image launches on, so a kernel reads a slot's buffer directly and a free (which synchronizes that context) waits
for every queued launch that reads it.

`seq_member_store_v1` is a transport entry with the x_prep numbering and signature (x_prep/transport_abi.mojo;
bindings/arena_cache_run.mojo ARENA_STORE_PUT / FREE / TAKE_ROWS / LIVE), published as entry 1 of the image's
x_sequence table (core/xseq_entry_table.mojo `xseq_store_ref`). The program operations (XP_RUN*) are refused: this
store runs no x_prep program. Device image only; the host image publishes no store and its members answer no
row index.

Serial by nature: each operation is one slot bookkeeping update or one device gather on one context.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_store import DeviceStore
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.shared_slot import SharedSlot
from sequence.exec_device import sequence_ctx
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS, XP_DEV_LIVE, xp_validate_call

comptime _STORE_NAME = (
    "MojolearnSeqMemberStoreIdenticalV2" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojolearnSeqMemberStoreFastV2"
)
comptime SEQ_MEMBER_STORE = SharedSlot[DeviceStore, _STORE_NAME, 3]


def seq_member_store_ptr(slot_id: Int, need_words: Int) raises -> Int:
    """The device address of store slot `slot_id` (at least `need_words`
    words, refused otherwise), for a launch on the shared sequence context."""
    return Int(SEQ_MEMBER_STORE.get_or_create_ptr()[].ptr(slot_id, need_words))


def _store_op(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    if op == XP_DEV_PUT:
        return SEQ_MEMBER_STORE.get_or_create_ptr()[].put(sequence_ctx(), a[0], p[0])
    if op == XP_DEV_FREE:
        SEQ_MEMBER_STORE.get_or_create_ptr()[].free(sequence_ctx(), p[0])
        return 0
    if op == XP_DEV_TAKE_ROWS:
        return SEQ_MEMBER_STORE.get_or_create_ptr()[].take_rows(sequence_ctx(), p[0], p[1], a[0], p[2])
    if op == XP_DEV_LIVE:
        return Int(SEQ_MEMBER_STORE.get_or_create_ptr()[].live_count())
    raise Error("x_sequence member store: operation " + String(op) + " is not a store operation")


def seq_member_store_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                        ff: Int, nf: Int, result: Int, context: Int,
                        sink: NativeDescriptorError) abi("C") -> Int:
    """The family store's transport entry (module note)."""
    try:
        xp_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = _store_op(operation, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
