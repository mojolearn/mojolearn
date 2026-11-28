# SPDX-License-Identifier: Apache-2.0
"""The one body of min-label connected components (lane sc-leftovers-1,
2026-11-28; AGENTS.md "One body per operation"): hooking one edge and
pointer jumping one node, `@always_inline` so every kernel that runs them
keeps its own loop, launch and buffers and calls these.

Callers: x_neighbors/iter_device.mojo `cc_hook_kernel`, `cc_hook_round_kernel`,
`cc_hook_dense_kernel`, `cc_jump_kernel` (weak connected components of a
graph); x_decomp/graph_device.mojo `xg_hook_kernel`, `xg_jump_kernel` (the
spectral embedding's components); umap/spectral_init.mojo `sc_label_hook_kernel`, `sc_label_jump_kernel`
(the spectral start's components). Before this file the hook was written
five times and the jump three times; the spectral start's hook lowered both
endpoints and both label entries (four atomics) where the components' hook
lowers the larger label's entry (one). Both reach the same fixed point
(every node labelled by the smallest node of its component, whatever the
thread order), so the labels, and the bits after them, do not change.

THE FIXED POINT. Labels only fall, each to a node of the same component no
larger than itself. A round is a hook over every edge then a jump over every
node, so every hook starts from compressed labels (each label a root,
lab[root] == root). Then an edge whose ends carry different labels has a root
`hi` with lab[hi] == hi > lo, and its hook lowers it: a round that lowers
nothing has equal labels across every edge, the fixed point.

A leaf: no launch, no allocation, no imports beyond the atomic.
"""
from std.atomic import Atomic

comptime CcLabelPtr = MutPointer[Int32, MutAnyOrigin]


@always_inline
def cc_hook_edge(lab: CcLabelPtr, u: Int, v: Int) -> Bool:
    """Hook the edge (u, v): the larger of its two labels' entries falls to
    the smaller (an integer atomic min). True when it could still lower an
    entry (a racing read only costs a spare round)."""
    var a = lab.unsafe_load(u)
    var b = lab.unsafe_load(v)
    if a == b:
        return False
    var lo = a if a < b else b
    var hi = b if a < b else a
    if lab.unsafe_load(Int(hi)) > lo:
        _ = Atomic[Int32].min(lab + Int(hi), lo)
        return True
    return False


@always_inline
def cc_jump_node(lab: CcLabelPtr, v: Int):
    """Pointer jumping: node v takes the root of its label chain (the chain
    strictly descends to a self-labelled node, so it ends)."""
    var p = Int(lab.unsafe_load(v))
    while Int(lab.unsafe_load(p)) != p:
        p = Int(lab.unsafe_load(p))
    lab.unsafe_store(v, Int32(p))
