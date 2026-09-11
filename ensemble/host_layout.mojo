# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2637: threaded host layout moves for the forest fit boundaries.

Lane forest-speed, 2026-09-11. The RandomForest and ExtraTrees fits took a
C-order float32 X, transposed it to column-major in ONE thread
(`_buffer.as_f32_colmajor` -> `transpose_f32`), and the binding then copied
that column-major block AGAIN into pinned host memory in one thread. The
isolation forest appended every cell into a `List`, appended every cell again
to transpose it, and wrote every cell a third time into pinned memory. At
1M x 220 each of those passes is 880 MB.

These helpers write the column-major bytes straight into the caller's
destination (the pinned staging buffer), split across the host thread pool.

WHY THE BITS CANNOT MOVE. Every element is read once and stored once at
`dst[c * nr + r] = src[r * nc + c]`; tasks own disjoint row ranges, so no
destination cell is written twice and none is computed from another. The
`ftz` variant applies the SAME scalar `checks.numerics.ftz` the serial upload
applied, per element. Threading changes only the order in which independent
stores happen. There is no arithmetic that could depend on that order.

THE PARALLELIZE TRAP (`gbdt/train.mojo`, the step-33 race): tasks capture raw
pointers only. Callers keep every owner alive past the call; the one owner
created here (`flags`) is transferred after the join.
"""

from max.algorithm import sync_parallelize
from std.memory import bitcast, memcpy

from checks.numerics import ftz

comptime HOST_LAYOUT_BLOCK_ROWS = 4096
"""Rows per task. At 220 float32 features a block reads 3.4 MB of source in
row order, so each task's strided writes stay inside its own rows."""

comptime HOST_LAYOUT_SERIAL_CELLS = 1 << 20
"""Below about a million cells the pool's dispatch costs more than it saves;
those inputs take the same loop in the calling thread."""

comptime HOST_COPY_CHUNK = 1 << 22
"""Elements per task of the flat threaded copy (16 MB of float32)."""


def colmajor_from_rowmajor_f32(
    src: MutPointer[Float32, MutUntrackedOrigin],
    dst: MutPointer[Float32, MutUntrackedOrigin],
    nr: Int,
    nc: Int,
):
    """`dst[c * nr + r] = src[r * nc + c]` for every cell (DEVIATION 2637).
    `src` and `dst` are valid, non-overlapping, `nr * nc` cells each."""
    if nr <= 0 or nc <= 0:
        return
    var n_blocks = (nr + HOST_LAYOUT_BLOCK_ROWS - 1) // HOST_LAYOUT_BLOCK_ROWS
    var sp = src
    var dp = dst

    def _block(b: Int) {imm sp, imm dp, imm nr, imm nc}:
        var r0 = b * HOST_LAYOUT_BLOCK_ROWS
        var r1 = min(r0 + HOST_LAYOUT_BLOCK_ROWS, nr)
        for c in range(nc):
            var dbase = c * nr
            for r in range(r0, r1):
                dp.unsafe_store(dbase + r, sp.unsafe_load(r * nc + c))

    if nr * nc < HOST_LAYOUT_SERIAL_CELLS or n_blocks == 1:
        for b in range(n_blocks):
            _block(b)
        return
    sync_parallelize(_block, n_blocks)


def copy_f32_threaded(
    src: MutPointer[Float32, MutUntrackedOrigin],
    dst: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
):
    """`dst[i] = src[i]` for `i` in `[0, n)`, in chunks across the pool
    (DEVIATION 2637). A pure byte move."""
    if n <= 0:
        return
    if n < HOST_LAYOUT_SERIAL_CELLS:
        memcpy(dest=dst, src=src, count=n)
        return
    var n_chunks = (n + HOST_COPY_CHUNK - 1) // HOST_COPY_CHUNK
    var sp = src
    var dp = dst

    def _chunk(k: Int) {imm sp, imm dp, imm n}:
        var i0 = k * HOST_COPY_CHUNK
        var i1 = min(i0 + HOST_COPY_CHUNK, n)
        memcpy(dest=dp + i0, src=sp + i0, count=i1 - i0)

    sync_parallelize(_chunk, n_chunks)


def colmajor_ftz_from_rowmajor_f32(
    src: MutPointer[Float32, MutUntrackedOrigin],
    dst: MutPointer[Float32, MutUntrackedOrigin],
    nr: Int,
    nc: Int,
) -> Bool:
    """`dst[c * nr + r] = ftz(src[r * nc + c])` for every cell, and True when
    every source cell is finite (DEVIATION 2637, the isolation forest's fit
    upload: DEVIATION 680's finite scan and DEVIATION 1942 row 10's flush in
    one threaded pass). On False the destination is fully written but the
    caller must refuse; it re-runs the serial named scan for the message."""
    if nr <= 0 or nc <= 0:
        return True
    var n_blocks = (nr + HOST_LAYOUT_BLOCK_ROWS - 1) // HOST_LAYOUT_BLOCK_ROWS
    var flags = List[UInt8](length=n_blocks, fill=UInt8(0))
    var fp = flags.unsafe_ptr()
    var sp = src
    var dp = dst

    def _block(b: Int) {imm sp, imm dp, imm fp, imm nr, imm nc}:
        var r0 = b * HOST_LAYOUT_BLOCK_ROWS
        var r1 = min(r0 + HOST_LAYOUT_BLOCK_ROWS, nr)
        var bad = False
        for c in range(nc):
            var dbase = c * nr
            for r in range(r0, r1):
                var v = sp.unsafe_load(r * nc + c)
                var bits = bitcast[DType.uint32](v)
                if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
                    bad = True
                dp.unsafe_store(dbase + r, ftz(v))
        if bad:
            fp.unsafe_store(b, UInt8(1))

    if nr * nc < HOST_LAYOUT_SERIAL_CELLS or n_blocks == 1:
        for b in range(n_blocks):
            _block(b)
    else:
        sync_parallelize(_block, n_blocks)
    var all_finite = True
    for b in range(n_blocks):
        if flags[b] != 0:
            all_finite = False
    _ = flags^
    return all_finite


def copy_ftz_f32_threaded(
    src: MutPointer[Float32, MutUntrackedOrigin],
    dst: MutPointer[Float32, MutUntrackedOrigin],
    n: Int,
) -> Bool:
    """`dst[i] = ftz(src[i])` for `i` in `[0, n)` in chunks across the pool,
    and True when every source cell is finite (DEVIATION 2638, the isolation
    forest's row-major query upload: DEVIATION 680's scan and DEVIATION 1942
    row 10's flush in one threaded pass, the same scalar `ftz` per element).
    On False the caller re-runs its serial named scan for the message."""
    if n <= 0:
        return True
    var n_chunks = (n + HOST_COPY_CHUNK - 1) // HOST_COPY_CHUNK
    var flags = List[UInt8](length=n_chunks, fill=UInt8(0))
    var fp = flags.unsafe_ptr()
    var sp = src
    var dp = dst

    def _chunk(k: Int) {imm sp, imm dp, imm fp, imm n}:
        var i0 = k * HOST_COPY_CHUNK
        var i1 = min(i0 + HOST_COPY_CHUNK, n)
        var bad = False
        for i in range(i0, i1):
            var v = sp.unsafe_load(i)
            var bits = bitcast[DType.uint32](v)
            if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
                bad = True
            dp.unsafe_store(i, ftz(v))
        if bad:
            fp.unsafe_store(k, UInt8(1))

    if n < HOST_LAYOUT_SERIAL_CELLS or n_chunks == 1:
        for k in range(n_chunks):
            _chunk(k)
    else:
        sync_parallelize(_chunk, n_chunks)
    var all_finite = True
    for k in range(n_chunks):
        if flags[k] != 0:
            all_finite = False
    _ = flags^
    return all_finite
