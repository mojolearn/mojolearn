# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`cast_elements` of the tree and label doors, split across the host's cores
(no Python import).

Lane A1 admit-trees-labels: bindings/classical_labels.mojo `labels_cast`
(the trees_* `trees_cast`) reached the base binding's `cast_elements` through
a Python call, one task (`_tasks = 1`, core/hotpath_impl.mojo). It now runs
the same typed kernel in process over contiguous element ranges, one range
per task (core/host_predict_threads.mojo, with its grain floor). The base
binding's own `cast_elements` export keeps its one task: it is GPU-binding
code, where the host-route fence refuses host threads.

    admitted_cast_elements(sa, sc, da, dc, count) -> Int
        dst[i] = <dc>(src[i]) for i in [0, count); 0, or 1 when an element
        is one the Python conversion refuses (`cast_elements`' contract)

Numerics: every element is converted on its own by the same conversion, so
the split moves no output bit; the refusal is the OR of the ranges'.
"""
from core.hotpath_impl import cast_elements
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk


@always_inline
def _width(code: Int) -> Int:
    """Bytes per element of a hotpath dtype code (HP_*)."""
    if code == 1 or code == 3:
        return 8
    if code == 0 or code == 2 or code == 4:
        return 4
    if code == 6 or code == 8:
        return 2
    return 1


def admitted_cast_elements(sa: Int, sc: Int, da: Int, dc: Int, count: Int) raises -> Int:
    # parallel: element ranges across host cores (lane A1: one task through the base binding)
    if count <= 0:
        return 0
    var tasks = host_predict_task_count_work(count, count)
    if tasks <= 1:
        return cast_elements(sa, sc, da, dc, count)
    # Whole SIMD runs (8 elements) per range but the last.
    var chunk = ((host_predict_chunk(count, tasks) + 7) // 8) * 8
    var ws = _width(sc)
    var wd = _width(dc)
    var flags = List[UInt8](length=tasks, fill=UInt8(0))
    var flag_ptr = rebind[MutPointer[UInt8, MutUntrackedOrigin]](flags.unsafe_ptr())

    def _range(t: Int) {imm sa, imm sc, imm da, imm dc, imm count, imm chunk, imm ws, imm wd, imm flag_ptr}:
        var lo = t * chunk
        if lo >= count:
            return
        var n = min(chunk, count - lo)
        try:
            if cast_elements(sa + lo * ws, sc, da + lo * wd, dc, n) != 0:
                flag_ptr.unsafe_store(t, UInt8(1))
        except:
            flag_ptr.unsafe_store(t, UInt8(1))

    host_parallelize(_range, tasks)
    var refused = 0
    for t in range(tasks):  # driver-loop: over the element tasks' flags (at most one per core)
        if flags[t] != UInt8(0):
            refused = 1
    _ = flags^
    return refused
