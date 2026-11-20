# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of core/compact_ops.mojo (lane n7-device-linalg-ops,
2026-11-08). No GPU import. Order-preserving compaction: out[k] is the k-th
flagged element in index order (an exclusive prefix sum of 0 / 1 flags is
one function, so the words are the device's). Tiles of CH_TILE elements:
the tiles' counts in parallel, one scan over the tile counts, the tiles'
scatters in parallel (a tile writes only its own output range).
"""
from cluster.host.host_cells import host_cells

comptime CH_TILE = 4096

comptime _HB = MutPointer[UInt8, MutAnyOrigin]
comptime _HI = MutPointer[Int32, MutAnyOrigin]
comptime _HF = MutPointer[Float32, MutAnyOrigin]


def _compact_starts(flags: _HB, n: Int) raises -> List[Int32]:
    """Each tile's first output slot, then the total (tiles + 1 words)."""
    var tiles = (n + CH_TILE - 1) // CH_TILE
    var starts = List[Int32](length=tiles + 1, fill=Int32(0))
    var sp = _HI(unsafe_from_address=Int(starts.unsafe_ptr()))

    def count(t: Int) {imm flags, imm sp, imm n}:
        var c = 0
        for i in range(t * CH_TILE, min((t + 1) * CH_TILE, n)):
            c += Int(flags.unsafe_load(i)) & 1
        sp.unsafe_store(t + 1, Int32(c))

    host_cells(count, tiles, CH_TILE)
    for t in range(tiles):  # the scan over tile counts (a prefix sum: one dependent chain)
        starts[t + 1] = starts[t + 1] + starts[t]
    return starts^


def compact_indices_host(flags: _HB, n: Int, dst: _HI) raises -> Int:
    """dst[0, count) = the flagged indices of [0, n) ascending; returns count."""
    if n <= 0:
        return 0
    var starts = _compact_starts(flags, n)
    var sp = _HI(unsafe_from_address=Int(starts.unsafe_ptr()))

    def scatter(t: Int) {imm flags, imm sp, imm dst, imm n}:
        var at = Int(sp.unsafe_load(t))
        for i in range(t * CH_TILE, min((t + 1) * CH_TILE, n)):
            if flags.unsafe_load(i) != UInt8(0):
                dst.unsafe_store(at, Int32(i))
                at += 1

    host_cells(scatter, len(starts) - 1, CH_TILE)
    return Int(starts[len(starts) - 1])


def compact_i32_host(src: _HI, flags: _HB, n: Int, dst: _HI) raises -> Int:
    """dst[0, count) = src's flagged elements in index order; returns count."""
    if n <= 0:
        return 0
    var starts = _compact_starts(flags, n)
    var sp = _HI(unsafe_from_address=Int(starts.unsafe_ptr()))

    def scatter(t: Int) {imm src, imm flags, imm sp, imm dst, imm n}:
        var at = Int(sp.unsafe_load(t))
        for i in range(t * CH_TILE, min((t + 1) * CH_TILE, n)):
            if flags.unsafe_load(i) != UInt8(0):
                dst.unsafe_store(at, src.unsafe_load(i))
                at += 1

    host_cells(scatter, len(starts) - 1, CH_TILE)
    return Int(starts[len(starts) - 1])


def compact_f32_host(src: _HF, flags: _HB, n: Int, dst: _HF) raises -> Int:
    """dst[0, count) = src's flagged elements in index order; returns count."""
    if n <= 0:
        return 0
    var starts = _compact_starts(flags, n)
    var sp = _HI(unsafe_from_address=Int(starts.unsafe_ptr()))

    def scatter(t: Int) {imm src, imm flags, imm sp, imm dst, imm n}:
        var at = Int(sp.unsafe_load(t))
        for i in range(t * CH_TILE, min((t + 1) * CH_TILE, n)):
            if flags.unsafe_load(i) != UInt8(0):
                dst.unsafe_store(at, src.unsafe_load(i))
                at += 1

    host_cells(scatter, len(starts) - 1, CH_TILE)
    return Int(starts[len(starts) - 1])
