# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane af-hdbscan2, `-D MOJOLEARN_HDB_LINKAGE_DEVICE` (FAST on Apple only):
`build_dendrogram_device`'s divide and conquer with the per-level hook loop
replaced by ONE lock-free union launch, so no level reads a flag back.

WHAT MAIN DOES (hierarchy/impl/cluster/detail/dendrogram_device.mojo). The
single linkage tree is already built on the device from the sorted MST
edges (both FAST MST arms emit them sorted by (weight key, lo, hi)): 17
levels at m = 100,000, and inside each level a loop of (memset flag, hook,
jump, 1-word readback, synchronize) until no lower-half edge joins two
roots, typically 2 to 3 iterations: ~40 waits over one-word flags.

WHAT THIS DOES. The components of a level's lower-half edges come from one
launch of a lock-free union-find (one thread per lower-half edge):

    loop: ra = root(a), rb = root(b); equal -> done;
          CAS par[max(ra, rb)] from itself to min(ra, rb); success -> done;
          else (another thread hooked that root first) retry from ra, rb.

A root is only ever hooked under a SMALLER root of another tree of the same
component, so (1) parent pointers only decrease and no cycle forms, (2) the
smallest label of a component is never hooked: after the launch every
component is one tree whose root is its smallest label. Main's converged
min-label hooking ends at the same root for every label (its parents also
only decrease, and it stops only when every edge's endpoints share a root,
which is the component minimum). Main's `_dd_jump_kernel` then compresses
every endpoint label to that root, and top / size / relabel read nothing
but those roots. The three outputs are therefore the same integers, bit
for bit, as main's (integer work only, no order dependence).

Every read of `par` inside the union launch is an atomic (relaxed) load,
so a retry sees the hook that defeated its CAS; a failed CAS means another
thread's hook landed, so the launch always finishes (lock-free, no thread
waits on another). No flag, no readback, no wait per level; the function
ends with no synchronize (the condense that follows is on the same queue).
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from hierarchy.impl.cluster.detail.dendrogram_device import (
    DD_TPB,
    _dd_claim_kernel,
    _dd_init_kernel,
    _dd_jump_kernel,
    _dd_out_kernel,
    _dd_own_kernel,
    _dd_relabel_kernel,
    _dd_size_kernel,
    _dd_top_kernel,
)

comptime I32P = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _du_gid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _du_blocks(n: Int) -> Int:
    return (n + DD_TPB - 1) // DD_TPB if n > 0 else 1


@always_inline
def _du_root(par: I32P, x0: Int32) -> Int32:
    """The root of x0 by relaxed atomic loads (inlined: Metal crashes on
    non-inlined pointer callees)."""
    var x = x0
    while True:
        var p = Atomic.load[ordering = Ordering.RELAXED](
            par.unsafe_offset(Int(x))
        )
        if p == x:
            return x
        x = p


def _du_union_kernel(la: I32P, lb: I32P, par: I32P, cnt: Int32, s: Int32):
    """One lower-half edge per thread: union by CAS, larger root under the
    smaller (module docstring)."""
    var i = _du_gid()
    var c = Int(cnt)
    var half = Int(s) // 2
    if i >= c or (i % Int(s)) >= half:
        return
    var a = la[i]
    var b = lb[i]
    while True:
        var ra = _du_root(par, a)
        var rb = _du_root(par, b)
        if ra == rb:
            return
        var hi = ra if ra > rb else rb
        var lo = rb if ra > rb else ra
        var expected = hi
        if Atomic.compare_exchange[
            success_ordering = Ordering.RELAXED,
            failure_ordering = Ordering.RELAXED,
            weak=True,  # Apple AIR has only weak CAS; a spurious fail retries
        ](par.unsafe_offset(Int(hi)), expected, lo):
            return
        a = ra
        b = rb


def build_dendrogram_union(
    ctx: DeviceContext,
    mut rows: DeviceBuffer[DType.int32],
    mut cols: DeviceBuffer[DType.int32],
    mut data: DeviceBuffer[DType.float32],
    nnz: Int,
    mut children: DeviceBuffer[DType.int32],
    mut out_delta: DeviceBuffer[DType.float32],
    mut out_size: DeviceBuffer[DType.int32],
) raises:
    """`build_dendrogram_device`'s three outputs, the same integers, with
    seven launches per level and no wait (module docstring)."""
    var cnt = nnz
    if cnt < 1:
        return
    var m = cnt + 1
    var n_nodes = 2 * m
    var la = ctx.enqueue_create_buffer[DType.int32](cnt)
    var lb = ctx.enqueue_create_buffer[DType.int32](cnt)
    var par = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var top = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var ssum = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var sz = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var stamp = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var owner = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var p_la = la.unsafe_ptr()
    var p_lb = lb.unsafe_ptr()
    var p_par = par.unsafe_ptr()
    var p_top = top.unsafe_ptr()
    var p_ssum = ssum.unsafe_ptr()
    var p_sz = sz.unsafe_ptr()
    var p_stamp = stamp.unsafe_ptr()
    var p_owner = owner.unsafe_ptr()
    var ge = _du_blocks(cnt)
    ctx.enqueue_function[_dd_init_kernel](
        rows.unsafe_ptr(), cols.unsafe_ptr(), p_la, p_lb, p_sz, p_stamp,
        Int32(cnt), Int32(n_nodes),
        grid_dim=(_du_blocks(n_nodes), 1, 1), block_dim=(DD_TPB, 1, 1),
    )
    var s = 1
    while s < cnt:
        s *= 2
    var lvl = 0
    while s >= 2:
        var S = Int32(s)
        var L = Int32(lvl)
        ctx.enqueue_function[_dd_claim_kernel](
            p_la, p_lb, p_par, p_top, p_ssum, p_stamp, p_owner, Int32(cnt),
            S, L, grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        ctx.enqueue_function[_dd_own_kernel](
            p_la, p_lb, p_owner, Int32(cnt), S,
            grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        ctx.enqueue_function[_du_union_kernel](
            p_la, p_lb, p_par, Int32(cnt), S,
            grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        ctx.enqueue_function[_dd_jump_kernel](
            p_la, p_lb, p_par, Int32(cnt), S,
            grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        ctx.enqueue_function[_dd_top_kernel](
            p_la, p_lb, p_par, p_top, p_ssum, p_sz, p_owner, Int32(cnt), S,
            grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        ctx.enqueue_function[_dd_size_kernel](
            p_la, p_lb, p_par, p_top, p_ssum, p_sz, p_owner, Int32(cnt), S,
            Int32(m), grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        ctx.enqueue_function[_dd_relabel_kernel](
            p_la, p_lb, p_par, p_top, p_stamp, Int32(cnt), S, L, Int32(m),
            grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
        )
        s //= 2
        lvl += 1
    ctx.enqueue_function[_dd_out_kernel](
        p_la, p_lb, data.unsafe_ptr(), p_sz, children.unsafe_ptr(),
        out_delta.unsafe_ptr(), out_size.unsafe_ptr(), Int32(cnt),
        grid_dim=(ge, 1, 1), block_dim=(DD_TPB, 1, 1),
    )
    # no synchronize: the condense is the next work on this queue (the
    # buffers below are released in queue order, as td_sort_pairs' are).
    _ = la^
    _ = lb^
    _ = par^
    _ = top^
    _ = ssum^
    _ = sz^
    _ = stamp^
    _ = owner^
