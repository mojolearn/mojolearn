# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE trees lane's fixed fold order for data-sized float64 sums
(lane cpu2-l5-trees, 2026-10-04).

The ensemble glue of xtrees/ops.mojo summed its float64 vectors in one
sequential chain over the rows, which no parallel device can reproduce. Every
such sum now runs in ONE order on every column (host, NVIDIA, AMD, Apple):

  CHUNK   rows fall in chunks of `FOLD_CHUNK`; a chunk's partial starts at
          +0.0 and adds its rows in index order (`run = run + v_i`);
  TREE    the m chunk partials are folded by a fixed pairwise tree: pass
          s = 1, 2, 4, ... while s < m: p[j] = p[j] + p[j + s] for every j
          with j % (2 s) == 0 and j + s < m; the answer is p[0] (+0.0 when
          there are no rows).

A record of W sums (several sums over the same rows, e.g. AdaBoost's error
and total) folds field by field with the same tree, p[j * W + q].

The device spelling is `fold_tree_pass_kernel` (one launch per pass, one
thread per (pair, field), float64 as binary64 words through
checks/soft_f64.mojo, the Apple GPU having no float64); the host spelling is
`fold_tree_host`. soft_f64's add is the correctly rounded IEEE add, so both
columns produce the same words. Old (sequential) bits are gone on every
column at once: only same bits across hardware matter.

This module imports nothing from xtrees/ops.mojo, so ops.mojo and the device
files may both import it."""
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.soft_f64 import sf64_add

#: rows per chunk partial (the same constant on every column).
comptime FOLD_CHUNK = 256
comptime FOLD_TPB = 256
comptime FOLD_MAX_BLOCKS = 65535


def fold_chunks(n: Int) -> Int:
    """The number of chunk partials of n rows (at least 1, so an empty sum is
    one +0.0 partial)."""
    return max(1, ceildiv(n, FOLD_CHUNK))


#: the most chunk-partial words a record fold may hold (2^22 words, 32 MB).
comptime FOLD_MAX_WORDS = 1 << 22


def fold_chunk_size(n: Int, w: Int) -> Int:
    """The chunk length of a w-field record fold over n rows: FOLD_CHUNK,
    doubled while the partials would exceed FOLD_MAX_WORDS (wide records,
    e.g. one field pair per tree node). A pure function of (n, w), so every
    column picks the same chunks."""
    var c = FOLD_CHUNK
    while ceildiv(max(n, 1), c) * max(w, 1) > FOLD_MAX_WORDS:
        c *= 2
    return c


def fold_tree_host(mut p: List[Float64], m: Int, w: Int):
    """In place: the TREE fold of m records of w float64 fields; the answers
    land in p[0 .. w)."""
    var s = 1
    while s < m:
        var j = 0
        while j + s < m:
            for q in range(w):
                p[j * w + q] = p[j * w + q] + p[(j + s) * w + q]
            j += 2 * s
        s *= 2


def fold_sum_host(v: List[Float64], n: Int) -> Float64:
    """CHUNK then TREE over v[0 .. n): the fixed-order sum."""
    var m = fold_chunks(n)
    var p = List[Float64](length=m, fill=0.0)
    for c in range(m):
        var run: Float64 = 0.0
        for i in range(c * FOLD_CHUNK, min((c + 1) * FOLD_CHUNK, n)):
            run = run + v[i]
        p[c] = run
    fold_tree_host(p, m, 1)
    return p[0]


def fold_tree_pass_kernel(p: MutPointer[UInt64, MutAnyOrigin], m: Int64, w: Int64, s: Int64):
    """One TREE pass: p[j] += p[j + s] (field by field) for j % (2 s) == 0,
    j + s < m. One thread per (pair, field), grid-stride; pairs are disjoint,
    so a pass has no race."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var pairs = (Int(m) + 2 * Int(s) - 1) // (2 * Int(s))
    var total = pairs * Int(w)
    while t < total:
        var pr = t // Int(w)
        var q = t - pr * Int(w)
        var j = pr * 2 * Int(s)
        if j + Int(s) < Int(m):
            var a = j * Int(w) + q
            var b = (j + Int(s)) * Int(w) + q
            p.unsafe_store(a, sf64_add(p.unsafe_load(a), p.unsafe_load(b)))
        t += stride


def fold_tree_device(ctx: DeviceContext, p: DeviceBuffer[DType.uint64], m: Int, w: Int) raises:
    """Enqueue the TREE passes over m records of w binary64 words in `p`
    (the answers land in p[0 .. w)). Nothing is synchronized here."""
    var s = 1
    while s < m:
        var pairs = ceildiv(m, 2 * s)
        ctx.enqueue_function[fold_tree_pass_kernel](
            # p is written in place through a read-only handle argument (box-run-2 compile fix)
            p.unsafe_ptr().unsafe_mut_cast[True](), Int64(m), Int64(w), Int64(s),
            grid_dim=max(1, min(ceildiv(pairs * w, FOLD_TPB), FOLD_MAX_BLOCKS)), block_dim=FOLD_TPB,
        )
        s *= 2
