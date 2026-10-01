# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lu_solve on the host as a block-interleaved walk of `lu_solve_col`'s
cells (lane neural-pass39, 2026-10-01).

The board's lu-factor and lu-solve cells are the two-call form
lu_solve(lu_factor(A), B) at 8,192 x 8,192 with 64 right-hand sides, and
their time is the solve: the device runs it as ONE THREAD PER COLUMN of B
(64 threads on the whole GPU), each a chain of n^2 dependent multiply-adds
at memory latency (the LU kernels themselves are ~0.3 s on an L40S or an
MI325X at 8,192 while the cells read 4.5-10 s). Here B is copied once into
a block-interleaved layout (the columns in blocks of W = HOST_FW, each
block's W columns row-interleaved), the swaps are row exchanges, and each
row's substitution advances the block's W chains together with
`identical_mul_add_simd` and `ftz_lanes` (the scalar fma and flush lane by
lane): every column's chain is `lu_solve_col`'s, the same operands in the
same order from the same start, the division per lane the cell's `div0`.
The blocks go over host tasks (a task's columns are its own; `lu` and
`piv` are read by every task and written by none), two blocks per loop so
their chains overlap. MOJOLEARN_XD_LU_SOLVE_SERIAL=1 keeps the host
column's serial loop; the device kit takes the walk for n >= 1,024
(MOJOLEARN_XD_LU_SOLVE_HOST=0/1 forces either).
"""
from std.os import getenv
from std.math import iota

from checks.numerics import ftz, identical_mul_add, identical_mul_add_simd
from core.host_lanes import F32V, HOST_FW, ftz_lanes, host_row_tasks
from core.host_parallel import host_parallelize
from x_decomp.cells import F32Ptr, I32Ptr, div0

comptime _FP = MutPointer[Float32, MutUntrackedOrigin]
comptime _W = HOST_FW
comptime _B32 = SIMD[DType.bool, _W]
comptime XD_LU_SOLVE_HOST_MIN = 1024


def xd_lu_solve_serial() -> Bool:
    """MOJOLEARN_XD_LU_SOLVE_SERIAL=1: the host column's column-by-column
    serial loop (the A/B arm); default the walk."""
    return String(getenv("MOJOLEARN_XD_LU_SOLVE_SERIAL")) == "1"


def xd_lu_solve_on_host(n: Int) -> Bool:
    """Whether the device kit runs lu_solve as the host walk: by default for
    n >= XD_LU_SOLVE_HOST_MIN; MOJOLEARN_XD_LU_SOLVE_HOST=0/1 forces either."""
    var v = String(getenv("MOJOLEARN_XD_LU_SOLVE_HOST"))
    if v == "0":
        return False
    if v == "1":
        return True
    return n >= XD_LU_SOLVE_HOST_MIN


@always_inline
def _pack_in(src: F32Ptr, rows: Int, cols: Int, mut dst: List[Float32]):
    var nb = (cols + _W - 1) // _W
    var dp = _FP(unsafe_from_address=Int(dst.unsafe_ptr()))
    for b in range(nb):
        var out = dp + b * rows * _W
        for i in range(rows):
            for l in range(_W):
                var j = b * _W + l
                out.unsafe_store(i * _W + l, src.unsafe_load(i * cols + j) if j < cols else Float32(0))


@always_inline
def _pack_out(src: List[Float32], rows: Int, cols: Int, dst: F32Ptr):
    var sp = _FP(unsafe_from_address=Int(src.unsafe_ptr()))
    for i in range(rows):
        for j in range(cols):
            dst.unsafe_store(i * cols + j, sp.unsafe_load((j // _W) * rows * _W + i * _W + (j % _W)))


@always_inline
def _swap_rows(blk: _FP, k: Int, p: Int):
    """The cell's swap of rows k and p of column c, for the block's W columns."""
    var t = blk.unsafe_load[width=_W](k * _W)
    blk.unsafe_store(k * _W, blk.unsafe_load[width=_W](p * _W))
    blk.unsafe_store(p * _W, t)


@always_inline
def _div_row(blk: _FP, i: Int, acc: F32V, d: Float32, mask: _B32):
    """b[i, c] = div0(acc[c], d) per lane; masked lanes keep their words."""
    var out = F32V(0)
    for l in range(_W):
        out[l] = div0(acc[l], d)
    blk.unsafe_store(i * _W, mask.select(out, blk.unsafe_load[width=_W](i * _W)))


@always_inline
def _store_row(blk: _FP, i: Int, acc: F32V, mask: _B32):
    blk.unsafe_store(i * _W, mask.select(acc, blk.unsafe_load[width=_W](i * _W)))


def _solve_blocks(lu: F32Ptr, piv: I32Ptr, base: _FP, n: Int, nrhs: Int, trans: Int, b0: Int, b1: Int):
    """`lu_solve_col` over the blocks [b0, b1) of the interleaved B, two
    blocks per loop."""
    var b = b0
    while b < b1:
        var two = b + 1 < b1
        var blkA = base + b * n * _W
        var blkB = base + (b + 1) * n * _W if two else blkA
        var laneA = iota[DType.int32, _W]() + Int32(b * _W)
        var laneB = laneA + Int32(_W)
        var mA = laneA.lt(Int32(nrhs))
        var mB = laneB.lt(Int32(nrhs)) if two else _B32(False)
        if trans != 0:
            for i in range(n):
                var accA = ftz_lanes(blkA.unsafe_load[width=_W](i * _W))
                var accB = ftz_lanes(blkB.unsafe_load[width=_W](i * _W))
                for j in range(i):
                    var l = F32V(-ftz(lu.unsafe_load(j * n + i)))
                    accA = ftz_lanes(identical_mul_add_simd[_W](l, ftz_lanes(blkA.unsafe_load[width=_W](j * _W)), accA))
                    accB = ftz_lanes(identical_mul_add_simd[_W](l, ftz_lanes(blkB.unsafe_load[width=_W](j * _W)), accB))
                var d = lu.unsafe_load(i * n + i)
                _div_row(blkA, i, accA, d, mA)
                if two:
                    _div_row(blkB, i, accB, d, mB)
            for ii in range(n):
                var i = n - 1 - ii
                var accA = ftz_lanes(blkA.unsafe_load[width=_W](i * _W))
                var accB = ftz_lanes(blkB.unsafe_load[width=_W](i * _W))
                for j in range(i + 1, n):
                    var l = F32V(-ftz(lu.unsafe_load(j * n + i)))
                    accA = ftz_lanes(identical_mul_add_simd[_W](l, ftz_lanes(blkA.unsafe_load[width=_W](j * _W)), accA))
                    accB = ftz_lanes(identical_mul_add_simd[_W](l, ftz_lanes(blkB.unsafe_load[width=_W](j * _W)), accB))
                _store_row(blkA, i, accA, mA)
                if two:
                    _store_row(blkB, i, accB, mB)
            for kk in range(n):
                var k = n - 1 - kk
                var p = Int(piv.unsafe_load(k))
                if p != k:
                    _swap_rows(blkA, k, p)
                    if two:
                        _swap_rows(blkB, k, p)
        else:
            for k in range(n):
                var p = Int(piv.unsafe_load(k))
                if p != k:
                    _swap_rows(blkA, k, p)
                    if two:
                        _swap_rows(blkB, k, p)
            for i in range(n):
                var accA = ftz_lanes(blkA.unsafe_load[width=_W](i * _W))
                var accB = ftz_lanes(blkB.unsafe_load[width=_W](i * _W))
                var row = lu + i * n
                for j in range(i):
                    var l = F32V(-ftz(row.unsafe_load(j)))
                    accA = ftz_lanes(identical_mul_add_simd[_W](l, ftz_lanes(blkA.unsafe_load[width=_W](j * _W)), accA))
                    accB = ftz_lanes(identical_mul_add_simd[_W](l, ftz_lanes(blkB.unsafe_load[width=_W](j * _W)), accB))
                _store_row(blkA, i, accA, mA)
                if two:
                    _store_row(blkB, i, accB, mB)
            for ii in range(n):
                var i = n - 1 - ii
                var accA = ftz_lanes(blkA.unsafe_load[width=_W](i * _W))
                var accB = ftz_lanes(blkB.unsafe_load[width=_W](i * _W))
                var row = lu + i * n
                for j in range(i + 1, n):
                    var l = F32V(-ftz(row.unsafe_load(j)))
                    accA = ftz_lanes(identical_mul_add_simd[_W](l, ftz_lanes(blkA.unsafe_load[width=_W](j * _W)), accA))
                    accB = ftz_lanes(identical_mul_add_simd[_W](l, ftz_lanes(blkB.unsafe_load[width=_W](j * _W)), accB))
                var d = lu.unsafe_load(i * n + i)
                _div_row(blkA, i, accA, d, mA)
                if two:
                    _div_row(blkB, i, accB, d, mB)
        b += 2 if two else 1


def lu_solve_host_rows(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int, trans: Int):
    """`lu_solve_serial` as the block-interleaved walk (the module docstring)."""
    if n <= 0 or nrhs <= 0:
        return
    var nb = (nrhs + _W - 1) // _W
    var bt = List[Float32](unsafe_uninit_length=max(nb * _W * n, 1))
    _pack_in(b, n, nrhs, bt)
    var base = _FP(unsafe_from_address=Int(bt.unsafe_ptr()))
    var tasks = host_row_tasks(nb, 2 * n * n)
    var chunk = (nb + tasks - 1) // tasks
    # two blocks a loop: an even chunk keeps the pairs inside a task
    if chunk % 2 == 1 and chunk < nb:
        chunk += 1
    def _run(task: Int) {imm lu, imm piv, imm base, imm n, imm nrhs, imm trans, imm chunk, imm nb}:
        var b0 = task * chunk
        var b1 = min(b0 + chunk, nb)
        if b1 > b0:
            _solve_blocks(lu, piv, base, n, nrhs, trans, b0, b1)
    var used = (nb + chunk - 1) // chunk
    if used <= 1:
        _run(0)
    else:
        host_parallelize(_run, used)
    _pack_out(bt, n, nrhs, b)
    _ = bt^
