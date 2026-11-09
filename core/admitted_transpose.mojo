# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The parallel tiled transpose of admitted input words (no Python import).

Lane A1 admit-trees-labels (docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md
section 3.3): the tree and label doors (bindings/classical_labels.mojo
`labels_f32_view`) asked for an F-order float32 X (or a C-order X of an
F-order source) through a Python call into the base binding's
`transpose_f32` / `cast_colmajor_f64_to_f32` (bindings/_mojolearn.mojo),
which uploaded X to the device, transposed it there and downloaded it. They
now call this in process.

    admitted_transpose_to_f32[S](src, dst, nr, nc)
        dst[c * nr + r] = Float32(src[r * nc + c]) for the C-contiguous
        nr x nc block of Scalar[S] at `src` (S = float32: a pure move;
        S = float64: the one narrowing per element). Blocks of
        _AT_TILE_ROWS source rows are split into contiguous tasks across the
        host's cores (core/host_predict_threads.mojo, with its grain floor).

Numerics: each element is written once, to its own cell, by the same single
cast `tiled_transpose_to_f32` (core/mojolearn_shell_impl.mojo) and the device
transpose make, so neither the tile shape nor the split changes an output
bit.
"""
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime _AT_TILE_ROWS = 128
comptime _AT_TILE_COLS = 64
"""The tile of `tiled_transpose_to_f32`: 64 KB of float64 in, 32 KB of
float32 out, so the strided side of the transpose hits a core's L1."""


def admitted_transpose_to_f32[S: DType](src: Int, dst: Int, nr: Int, nc: Int):
    """The column-major float32 layout of the C-contiguous nr x nc block of
    Scalar[S] at byte address `src`, written to `dst` (no overlap). The same
    call turns a column-major block into C order: pass nr = cols,
    nc = rows. Every cell of `dst` is written, so it may be uninitialised."""
    # parallel: row bands of tiles across host cores (lane A1: the device round trip through the base binding)
    if nr <= 0 or nc <= 0:
        return
    var sp = MutPointer[Scalar[S], MutUntrackedOrigin](unsafe_from_address=src)
    var dp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst)
    var bands = (nr + _AT_TILE_ROWS - 1) // _AT_TILE_ROWS
    var tasks = host_predict_task_count_work(bands, nr * nc)
    var chunk = host_predict_chunk(bands, tasks)

    def _bands(t: Int) {imm sp, imm dp, imm nr, imm nc, imm chunk}:
        var r0 = t * chunk * _AT_TILE_ROWS
        var stop = min((t + 1) * chunk * _AT_TILE_ROWS, nr)
        while r0 < stop:
            var r1 = min(r0 + _AT_TILE_ROWS, stop)
            var c0 = 0
            while c0 < nc:
                var c1 = min(c0 + _AT_TILE_COLS, nc)
                # One column at a time inside the tile: contiguous writes,
                # strided reads within the tile's rows (already in cache).
                for c in range(c0, c1):
                    var base = c * nr
                    for r in range(r0, r1):
                        dp.unsafe_store(base + r, sp.unsafe_load(r * nc + c).cast[DType.float32]())
                c0 = c1
            r0 = r1

    if tasks <= 1:
        _bands(0)
    else:
        host_parallelize(_bands, tasks)
