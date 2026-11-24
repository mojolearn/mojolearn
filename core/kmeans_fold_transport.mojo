# SPDX-License-Identifier: Apache-2.0
"""The KMeans fold image's store as a scope transport entry (lane vdf-scope-cluster, 2026-11-24).

The `resident()` scope keeps one cache per binding family on the family's own store, reached through a transport
entry with the x_prep transport numbering (bindings/arena_cache_run.mojo `ARENA_STORE_*`, lane vdf-scope-run). The
KMeans fold kernels (cluster/kmeans_fold.mojo, `KMEANS_CAP_FOLD`) read model selection's identical fold store
(core/msel_device_x.mojo `MSEL_FOLD_STORE`, context `msel_device_x_context()`), so this entry is that store:
  ARENA_STORE_PUT        a[0] host address, p[0] 4-byte words  -> a fresh slot id (bytes = 4 * words; waits)
  ARENA_STORE_FREE       p[0] slot id                         -> 0 (waits for the context, drops the buffer)
  ARENA_STORE_LIVE                                            -> the store's live slots
  ARENA_STORE_TAKE_ROWS  refused in words: the KMeans fold kernels gather the rows themselves, in the launch that
                         reads them (no separate gathered slot).
No kernel, no numerics: words are copied, never converted. Serial by nature: each op is one store bookkeeping
update and one copy on one context.
"""
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from core.entry_table import EntryRef, entry_ref_word
from core.msel_device_x import msel_fold_store, msel_device_x_context
from x_prep.transport_abi import XP_OP_COUNT, XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS, XP_DEV_LIVE, xp_validate_call

#: The scope cache's family prefix of this store.
comptime KMEANS_FOLD_PREFIX = "kmeans_fold"


def _fold_op(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    if op == XP_DEV_PUT:
        if a[0] == 0 or p[0] < 1:
            raise Error("kmeans fold store: a put needs at least one word at a non-null address")
        return msel_fold_store()[].put(msel_device_x_context(), a[0], 4 * p[0])
    if op == XP_DEV_FREE:
        msel_fold_store()[].free(msel_device_x_context(), p[0])
        return 0
    if op == XP_DEV_LIVE:
        return msel_fold_store()[].live_count()
    if op == XP_DEV_TAKE_ROWS:
        raise Error("kmeans fold store: no take_rows (the KMeans fold kernels gather their rows in the launch)")
    raise Error("kmeans fold store: unknown transport operation " + String(op))


def kmeans_fold_transport_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                             ff: Int, nf: Int, result: Int, context: Int,
                             sink: NativeDescriptorError) abi("C") -> Int:
    try:
        xp_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = _fold_op(operation, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


def kmeans_fold_store_ref(mode: Int, vendor: Int) -> EntryRef:
    """The transport entry of the fold store for a member route's (mode, vendor): no table, the function itself
    (it lives in every image that carries the KMeans fold entry)."""
    return EntryRef(kmeans_fold_transport_v1, entry_ref_word(mode, vendor, XP_OP_COUNT))
