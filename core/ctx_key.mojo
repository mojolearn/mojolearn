"""Device-context key for process caches (lane/fam2-lm, 2026-10-04).

A process cache that holds device or pinned host buffers must not hand a
buffer created on one `DeviceContext` to a launch on another: the two-GPU
drivers (`DeviceContext(device_id=rank)`) run two contexts in one process.
`ctx_cache_key(ctx)` is the key such a cache indexes its entries by: the
context itself (its runtime handle's address; lane/review-fixes, it was
the device id).

`ctx_cache_slot(ids, ctx)` finds the key in a cache's `ids` list and returns
its index, or -1 when the context has no entry yet (the cache then appends
its buffers and the key, in that order, so a raise in an allocation leaves
no key without buffers).
"""
from max.gpu.host import DeviceContext

# TOMBSTONE: MOJOLEARN_IDN_CACHE_CTX_KEY_OFF (old path: key 0 for every context, the one-context first-pass caches; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at afc5c11e5b.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_CACHE_CTX_KEY_OFF.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (its term here, the same key-0 old path) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at afc5c11e5b.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-03-base.patch.


def ctx_cache_key(ctx: DeviceContext) raises -> Int:
    """lane/review-fixes: the key is the CONTEXT, not its device id. Two
    contexts on one GPU have two streams, and a cached workspace used
    without a wait (the Mamba GEMM workspace) must never be shared between
    them. The key is the address of the context's runtime handle: copies of
    one context share it, a second `DeviceContext` on the same GPU does not,
    and a cached buffer keeps its context alive, so an address is never
    reused while a cache holds an entry under it."""
    # TOMBSTONE: MOJOLEARN_IDN_CACHE_CTX_KEY_DEVICE_ID (opt-in device-id key instead of the context handle; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at b789195239.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CACHE_CTX_KEY_DEVICE_ID.patch; record in docs/TOMBSTONES.md.
    # _handle is Optional[pointer] in Mojo 1.0 (box-run-2 compile fix).
    return Int(ctx._handle.value())


def ctx_cache_slot(ids: List[Int], ctx: DeviceContext) raises -> Int:
    var key = ctx_cache_key(ctx)
    for i in range(len(ids)):  # small-loop(ids: cached device contexts): one entry per live device context
        if ids[i] == key:
            return i
    return -1
