"""Device-context key for process caches (lane/fam2-lm, 2026-10-04).

A process cache that holds device or pinned host buffers must not hand a
buffer created on one `DeviceContext` to a launch on another: the two-GPU
drivers (`DeviceContext(device_id=rank)`) run two contexts in one process.
`ctx_cache_key(ctx)` is the key such a cache indexes its entries by: the
context's device id. Default ON; `-D MOJOLEARN_IDN_CACHE_CTX_KEY_OFF` (or
`-D MOJOLEARN_IDN_ALL_OFF`) returns 0 for every context, the one-context
form the first-pass caches had.

`ctx_cache_slot(ids, ctx)` finds the key in a cache's `ids` list and returns
its index, or -1 when the context has no entry yet (the cache then appends
its buffers and the key, in that order, so a raise in an allocation leaves
no key without buffers).
"""
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext

comptime IDN_CACHE_CTX_KEY = not (
    is_defined["MOJOLEARN_IDN_CACHE_CTX_KEY_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


def ctx_cache_key(ctx: DeviceContext) raises -> Int:
    comptime if IDN_CACHE_CTX_KEY:
        return Int(ctx.id())
    else:
        return 0


def ctx_cache_slot(ids: List[Int], ctx: DeviceContext) raises -> Int:
    var key = ctx_cache_key(ctx)
    for i in range(len(ids)):
        if ids[i] == key:
            return i
    return -1
