# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IsotonicRegression's fit on the host column (the CPU-only install and the
host binding, bindings/_mojolearn_x_linear_host.mojo).

cpu-gpu-cleanup c-linear (2026-10-02): this entry and its sort on host
threads moved here out of x_linear/isotonic.mojo, which the device binding
imports, so no GPU-reachable module runs host threads. The device fit is
x_linear/device.mojo `_iso_fit_grid`; both produce the same permutation (the
order (x, y, row) is total) and then run the same statements
(`iso_fit_sorted` here, its grid kernels there).
"""
from x_linear.ops import FP, IP, ld, st, ldi, sti
from x_linear.isotonic import iso_merge_passes, iso_fit_sorted
from core.host_lanes import host_row_tasks
from core.host_parallel import host_parallelize


def _iso_sort_host(x: FP, y: FP, perm: IP, tmp: IP, nk: Int):
    """perm[0, nk) sorted by (x, y, index): chunks over host tasks, then the
    merge passes over the sorted chunks."""
    var tasks = host_row_tasks(nk, 64)
    if tasks > 64:
        tasks = 64
    if tasks <= 1 or nk < 4096:
        iso_merge_passes(x, y, perm, tmp, 0, nk, 1)
        return
    var chunk = (nk + tasks - 1) // tasks
    def _sort_chunk(task: Int) {imm x, imm y, imm perm, imm tmp, imm nk, imm chunk}:
        var lo = task * chunk
        var hi = lo + chunk
        if hi > nk:
            hi = nk
        if lo < hi:
            iso_merge_passes(x, y, perm, tmp, lo, hi, 1)
    host_parallelize(_sort_chunk, tasks)
    iso_merge_passes(x, y, perm, tmp, 0, nk, chunk)


def isotonic_fit_host(x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """x: n values (d == 1), ANY order; y: targets n | weights n (when
    ip[3] is set; otherwise weights 1).
    ip: [increasing, has_y_min, has_y_max, has_weights]; fp: [y_min, y_max].
    res: m | X_min | X_max | xs n | ys n.
    fw: ux n | uy n | uw n | sx n | sy n | sw n.  iw: target n | perm n | tmp n.
    lane/neural-pass70 (2026-10-01): the rows with a positive weight are
    sorted here by (x, y, row), the stable sort by (x, y) the Python layer
    did over a list of a million pairs, then pool-adjacent-violators over
    the sorted copies: the same order, the same chains, the same bits."""
    var has_w = ldi(ip, 3) != 0
    var perm = iw + n
    var tmp = iw + 2 * n
    var nk = 0
    for i in range(n):
        if not has_w or ld(y, n + i) > Float32(0):
            sti(perm, nk, i)
            nk += 1
    if nk < 1:
        st(res, 0, Float32(0))
        st(res, 1, Float32(0))
        st(res, 2, Float32(0))
        return
    _iso_sort_host(x, y, perm, tmp, nk)
    iso_fit_sorted(x, y, n, ip, fp, res, fw, iw, perm, nk)
