# SPDX-License-Identifier: Apache-2.0
"""One family image's device store as a `resident()` scope transport entry (lane vdf-scope-cluster2b, 2026-11-24).

The scope keeps one cache per binding family on that family's own store, reached through a transport entry with
the x_prep transport numbering (bindings/arena_cache_run.mojo `ARENA_STORE_*`, lane vdf-scope-run;
x_prep/transport_abi.mojo `XP_DEV_*`). A family whose kernels run on their own DeviceContext (mixture, density,
spectral, manifold) keeps a core/device_store.mojo `DeviceStore` beside that context, and its transport entry is
three lines over `family_store_op`:

  ARENA_STORE_PUT        a[0] host address, p[0] 4-byte words        -> a fresh slot id (waits for the copy)
  ARENA_STORE_FREE       p[0] slot id                                -> 0 (waits for the context, drops the slot)
  ARENA_STORE_TAKE_ROWS  a[0] int64 row numbers, p[0] source slot id, p[1] words a row, p[2] row count
                         -> a fresh slot holding those rows, gathered ON THE DEVICE (`DeviceStore.take_rows`)
  ARENA_STORE_LIVE                                                   -> the store's live slots
  ARENA_STORE_READ       a[0] host address, p[0] slot id, p[1] words -> 0 (the slot's first words home; waits)

No numerics: words are copied, never converted, so a slot holds exactly the bytes the host handed over and a
taken slot exactly the bytes a host gather of the same rows would hold. Serial by nature: each op is one store
bookkeeping update and one copy or one gather launch on one context (the gather itself is a parallel kernel).
"""
from max.gpu.host import DeviceContext
from core.device_store import DeviceStore
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS, XP_DEV_LIVE, XP_DEV_READ, xp_validate_call


def family_store_op(mut store: DeviceStore, ctx: DeviceContext, family: String, op: Int, a: List[Int],
                    p: List[Int]) raises -> Int:
    """One transport operation on `store` (made on `ctx`); `family` names the store in a refusal."""
    if op == XP_DEV_PUT:
        if a[0] == 0 or p[0] < 1:
            raise Error(family + " store: a put needs at least one word at a non-null address")
        return store.put(ctx, a[0], p[0])
    if op == XP_DEV_FREE:
        store.free(ctx, p[0])
        return 0
    if op == XP_DEV_TAKE_ROWS:
        return store.take_rows(ctx, p[0], p[1], a[0], p[2])
    if op == XP_DEV_LIVE:
        return Int(store.live)
    if op == XP_DEV_READ:
        # Lane vdf-scope-cluster8, an addition (bindings/arena_cache_run.mojo `arena_store_read`): a slot a fitted
        # state keeps read home once (the export of its training rows block); x_prep's own read, same words.
        if p[1] < 0 or (p[1] > 0 and a[0] == 0):
            raise Error(family + " store: invalid read buffer")
        store.read(ctx, p[0], a[0], p[1])
        return 0
    raise Error(family + " store: transport operation " + String(op) + " is not a store operation")


def family_store_entry(mut store: DeviceStore, ctx: DeviceContext, family: String, operation: Int, aa: Int,
                       na: Int, pp: Int, np: Int, nf: Int, result: Int, context: Int,
                       sink: NativeDescriptorError) -> Int:
    """The body of a family's `abi("C")` transport entry: validate the spans (the x_prep schema), run the op,
    write the result word; 0, or 1 with the refusal reported to `sink`."""
    try:
        xp_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = family_store_op(
            store, ctx, family, operation, a, p)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
