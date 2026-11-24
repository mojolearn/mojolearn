# SPDX-License-Identifier: Apache-2.0
"""The model-selection fold store as a scope transport entry (lane vdf-scope-neural2, 2026-11-24): the device store
behind the PCA / TruncatedSVD members' THE RESIDENT ROUTE (core/member_abi.mojo MEMBER_CAN_ROW_INDEX /
MEMBER_CAN_SCOPE).

The resident-fold entries of PCA and TruncatedSVD (core/pca_native_fold.mojo `pca_fold_native_v1`,
core/tsvd_native_fold.mojo `tsvd_fold_native_v1`) already fit and transform a fold named by a base id and an int64
index id of the identical model-selection store (core/msel_device_x.mojo MSEL_FOLD_STORE), gathering the rows on the
device; the estimators' doors call them for `DeviceRows` input. `msel_fold_store_v1` hands that store to a member as
a transport entry with the x_prep numbering (x_prep/transport_abi.mojo; bindings/arena_cache_run.mojo
ARENA_STORE_PUT / FREE / LIVE), so the caller's `resident()` scope keeps the whole X there ONCE (the scope's cache of
the family "msel_fold") and a member puts only each fold's index words. The store counts bytes; the transport counts
4-byte words. TAKE_ROWS is refused: the fold entries gather from (base, index) themselves.

The store and its context (`msel_device_x_context`) are the identical base binding's, named process-wide, so this
entry is published once by the GPU base binding (core/mojolearn_run.mojo, core/estimators_entry_table.mojo family
ESTIMATORS_TABLE_DECOMP_FOLD, beside the two fold entries); lane vdf-scope-neural6: a `fast` build publishes it in its
own (mode 0) table too, naming the same identical store and context, so a `fast` member's fold entries share it.

Serial by nature: each operation is one slot bookkeeping update on one context.
"""
from core.msel_device_x import msel_fold_store, msel_device_x_context
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS, XP_DEV_LIVE, xp_validate_call


def _fold_store_op(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    var store = msel_fold_store()
    if op == XP_DEV_PUT:
        return store[].put(msel_device_x_context(), a[0], p[0] * 4)
    if op == XP_DEV_FREE:
        store[].free(msel_device_x_context(), p[0])
        return 0
    if op == XP_DEV_LIVE:
        return store[].live_count()
    if op == XP_DEV_TAKE_ROWS:
        raise Error("model-selection fold store: the fold entries gather (base, index) themselves; no take_rows")
    raise Error("model-selection fold store: operation " + String(op) + " is not a store operation")


def msel_fold_store_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                       ff: Int, nf: Int, result: Int, context: Int,
                       sink: NativeDescriptorError) abi("C") -> Int:
    """The fold store's transport entry (module note)."""
    try:
        xp_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = _fold_store_op(operation, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
