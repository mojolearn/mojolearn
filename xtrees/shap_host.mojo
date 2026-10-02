# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TreeSHAP on the host, for the CPU-only binding (bindings/_mojolearn_x_trees_host.mojo)
and the CPU verification column: the units of xtrees/shap.mojo in the stage
order of xtrees/shap_device.mojo, so its bits are the GPU columns' bits.
The unit loops are split over host tasks; every unit writes its own cells
(the integer atomics of the cover and the meta words are order free), so the
task count moves no bit. GPU installs never import this file."""
from core.host_lanes import host_row_tasks
from core.host_parallel import host_parallelize
from xtrees.shap import (
    F32P, I32P, SHAP_META_BAD, SHAP_META_WORDS,
    shap_parent_unit, shap_depth_unit, shap_cover_unit, shap_slot_unit, shap_ev_part_unit, shap_ev_fold_unit,
    shap_tree_unit, shap_fold_unit,
)

comptime _BUF_FLOATS = 32 * 1024 * 1024


struct _HostForest(Movable):
    var parent: List[Int32]
    var slot: List[Int32]
    var meta: List[Int32]

    def __init__(out self, forest: List[Int], d: Int, n_trees: Int, n_nodes: Int):
        self.parent = List[Int32](length=max(n_nodes, 1), fill=-1)
        self.slot = List[Int32](length=max(n_trees * d, 1), fill=0)
        self.meta = List[Int32](length=SHAP_META_WORDS, fill=0)
        var offsets = I32P(unsafe_from_address=forest[0])
        var colid = I32P(unsafe_from_address=forest[1])
        var left = I32P(unsafe_from_address=forest[3])
        var parent = self.parent.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var slot = self.slot.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var meta = self.meta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        for u in range(n_nodes):
            shap_parent_unit(u, offsets, n_trees, colid, left, d, parent, slot, meta)
        for t in range(n_trees):
            shap_slot_unit(t, d, slot, meta)

    def ptr(mut self, which: Int) -> I32P:
        if which == 0:
            return self.parent.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        if which == 1:
            return self.slot.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        return self.meta.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _tasks_over[FuncType: def(Int) -> None](ref func: FuncType, units: Int, work: Int):
    """func(u) for u in [0, units), split over host tasks."""
    var tasks = host_row_tasks(units, work)
    var chunk = (units + tasks - 1) // max(tasks, 1)
    def _run(task: Int) {imm func, imm chunk, imm units}:
        for u in range(task * chunk, min(units, (task + 1) * chunk)):
            func(u)
    if tasks <= 1:
        _run(0)
    else:
        host_parallelize(_run, tasks)


def shap_prepare(forest: List[Int], tscale: Int, bg: Int, cover_out: Int, ev: Int, meta_out: Int, nb: Int, d: Int,
                 n_trees: Int, k: Int, n_nodes: Int) raises:
    """xtrees/shap_device.mojo `shap_prepare`, on the host."""
    var fo = _HostForest(forest, d, n_trees, n_nodes)
    var offsets = I32P(unsafe_from_address=forest[0])
    var colid = I32P(unsafe_from_address=forest[1])
    var quesval = F32P(unsafe_from_address=forest[2])
    var left = I32P(unsafe_from_address=forest[3])
    var leaves = F32P(unsafe_from_address=forest[4])
    var ts = F32P(unsafe_from_address=tscale)
    var cover = I32P(unsafe_from_address=cover_out)
    var meta = fo.ptr(2)
    var parent = fo.ptr(0)
    for u in range(n_nodes):
        cover[unsafe_offset=u] = 0
    for u in range(n_nodes):
        shap_depth_unit(u, offsets, n_trees, parent, meta)
    var x = F32P(unsafe_from_address=bg)
    for u in range(n_trees * nb):
        shap_cover_unit(u, nb, offsets, colid, quesval, left, x, d, cover, meta)
    var part = List[Float32](length=max(n_trees * k, 1), fill=0.0)
    var pp = part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for u in range(n_trees * k):
        shap_ev_part_unit(u, k, offsets, left, leaves, cover, ts, pp)
    var e = F32P(unsafe_from_address=ev)
    for j in range(k):
        shap_ev_fold_unit(j, n_trees, k, pp, e)
    if fo.meta[SHAP_META_BAD] != 0:
        raise Error("x_trees tree_shap: malformed tree")
    var mo = I32P(unsafe_from_address=meta_out)
    for i in range(SHAP_META_WORDS):
        mo[unsafe_offset=i] = fo.meta[i]
    _ = part^
    _ = fo^


def tree_shap_values(forest: List[Int], tscale: Int, cover_in: Int, x: Int, phi: Int, n: Int, d: Int, n_trees: Int,
                     k: Int, n_nodes: Int, slots: Int, width: Int) raises:
    """xtrees/shap_device.mojo `tree_shap_values`, on the host."""
    var fo = _HostForest(forest, d, n_trees, n_nodes)
    var offsets = I32P(unsafe_from_address=forest[0])
    var colid = I32P(unsafe_from_address=forest[1])
    var quesval = F32P(unsafe_from_address=forest[2])
    var left = I32P(unsafe_from_address=forest[3])
    var leaves = F32P(unsafe_from_address=forest[4])
    var ts = F32P(unsafe_from_address=tscale)
    var cover = I32P(unsafe_from_address=cover_in)
    var parent = fo.ptr(0)
    var slot = fo.ptr(1)
    var meta = fo.ptr(2)
    var out = F32P(unsafe_from_address=phi)
    var sl = max(slots, 1)
    var rows = max(1, min(n, _BUF_FLOATS // (n_trees * sl * k)))
    var buf = List[Float32](length=max(rows * n_trees * sl * k, 1), fill=0.0)
    var bp = buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var nodes = Int(offsets[unsafe_offset=n_trees])
    var r0 = 0
    while r0 < n:
        var rc = min(rows, n - r0)
        var xp = F32P(unsafe_from_address=x + r0 * d * 4)
        comptime for wi in range(6):
            comptime W = 8 << wi
            if width == W:
                def _tree(u: Int) {imm rc, imm d, imm k, imm sl, imm offsets, imm colid, imm quesval, imm left,
                                   imm leaves, imm parent, imm cover, imm ts, imm slot, imm xp, imm bp, imm meta}:
                    shap_tree_unit[W](u, rc, d, k, sl, offsets, colid, quesval, left, leaves, parent, cover, ts,
                                      slot, xp, bp, meta)
                _tasks_over(_tree, n_trees * rc, 64 * max(nodes // max(n_trees, 1), 1))
        def _fold(u: Int) {imm r0, imm rc, imm n_trees, imm d, imm k, imm sl, imm slot, imm bp, imm out}:
            shap_fold_unit(u, r0, rc, n_trees, d, k, sl, slot, bp, out)
        _tasks_over(_fold, d * k * rc, n_trees)
        r0 += rc
    if fo.meta[SHAP_META_BAD] != 0:
        raise Error("x_trees tree_shap: malformed tree")
    _ = buf^
    _ = fo^
