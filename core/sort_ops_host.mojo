# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of core/sort_ops.mojo (lane n7-device-linalg-ops,
2026-11-08). No GPU import.

`sort_op_f32_host` is the device's sort: the same keys (`sort_key_bits`),
the same stable LSD radix of 8-bit digits over FRS_TILE-element tiles
(core/fast_radix_sort.mojo): per pass the digit counts of every tile (the
tiles in parallel), one exclusive scan of the digit-major counts, then each
tile scatters its elements in input order (the tiles in parallel; a tile
writes only the slots its counts own). A stable sort by the full key is one
permutation, so the words are the device's.
"""
from std.memory import bitcast
from cluster.host.host_cells import host_cells

#: core/fast_radix_sort.mojo's FRS_TILE and FRS_BINS (not imported: that
#: module is the GPU's). The tile size moves no word (one permutation).
comptime FRS_TILE = 256
comptime FRS_BINS = 256

comptime _HF = MutPointer[Float32, MutAnyOrigin]
comptime _HI = MutPointer[Int32, MutAnyOrigin]
comptime _HU = MutPointer[UInt32, MutAnyOrigin]
comptime _HC = MutPointer[Int32, MutAnyOrigin]


@always_inline
def _key_bits(v: Float32, descending: Bool) -> UInt32:
    """core/sort_ops.mojo `sort_key_bits` (repeated here so this module has
    no GPU import; the two are one map)."""
    var bits = bitcast[DType.uint32](v)
    var key: UInt32
    if bits & UInt32(0x80000000) != UInt32(0):
        key = ~bits
    else:
        key = bits | UInt32(0x80000000)
    if descending:
        key = ~key
    return key


@always_inline
def _hu(l: List[UInt32]) -> _HU:
    return _HU(unsafe_from_address=Int(l.unsafe_ptr()))


def sort_op_f32_host(keys: _HF, order: _HI, n: Int, descending: Bool) raises:
    """order[r] = the input index of the r-th key of keys[0, n) in the
    stable ascending (or descending) order of `sort_key_bits`."""
    if n <= 0:
        return
    var n_tiles = (n + FRS_TILE - 1) // FRS_TILE
    var m = FRS_BINS * n_tiles
    var kl = List[UInt32](length=n, fill=UInt32(0))
    var vl = List[UInt32](length=n, fill=UInt32(0))
    var tkl = List[UInt32](length=n, fill=UInt32(0))
    var tvl = List[UInt32](length=n, fill=UInt32(0))
    var cl = List[Int32](length=m, fill=Int32(0))
    var k0 = _hu(kl)
    var v0 = _hu(vl)
    var k1 = _hu(tkl)
    var v1 = _hu(tvl)
    var cp = _HC(unsafe_from_address=Int(cl.unsafe_ptr()))

    def keyed(i: Int) {imm keys, imm k0, imm v0, imm descending}:
        k0.unsafe_store(i, _key_bits(keys.unsafe_load(i), descending))
        v0.unsafe_store(i, UInt32(i))

    host_cells(keyed, n, 4)
    for p in range(4):  # driver-loop: over the four 8-bit digit passes
        var shift = UInt32(8 * p)
        var sk = k0 if p % 2 == 0 else k1
        var sv = v0 if p % 2 == 0 else v1
        var dk = k1 if p % 2 == 0 else k0
        var dv = v1 if p % 2 == 0 else v0

        def count_tile(t: Int) {imm sk, imm cp, imm shift, imm n, imm n_tiles}:
            for d in range(FRS_BINS):
                cp.unsafe_store(d * n_tiles + t, Int32(0))
            var lo = t * FRS_TILE
            var hi = min(lo + FRS_TILE, n)
            for i in range(lo, hi):
                var d = Int((sk.unsafe_load(i) >> shift) & UInt32(255))
                cp.unsafe_store(d * n_tiles + t, cp.unsafe_load(d * n_tiles + t) + Int32(1))

        host_cells(count_tile, n_tiles, 2 * FRS_TILE)
        # the exclusive scan of the digit-major counts (a prefix sum: one
        # dependent chain of m integer adds, m = n / FRS_TILE * 256)
        var run = Int32(0)
        for j in range(m):
            var c = cl[j]
            cl[j] = run
            run += c

        def scatter_tile(t: Int) {imm sk, imm sv, imm dk, imm dv, imm cp, imm shift, imm n, imm n_tiles}:
            var lo = t * FRS_TILE
            var hi = min(lo + FRS_TILE, n)
            for i in range(lo, hi):
                var key = sk.unsafe_load(i)
                var slot = Int((key >> shift) & UInt32(255)) * n_tiles + t
                var at = Int(cp.unsafe_load(slot))
                cp.unsafe_store(slot, Int32(at + 1))
                dk.unsafe_store(at, key)
                dv.unsafe_store(at, sv.unsafe_load(i))

        host_cells(scatter_tile, n_tiles, 4 * FRS_TILE)
    # four passes: the answer is back in (k0, v0)
    def out(r: Int) {imm v0, imm order}:
        order.unsafe_store(r, Int32(Int(v0.unsafe_load(r))))

    host_cells(out, n, 2)
    _ = kl^
    _ = vl^
    _ = tkl^
    _ = tvl^
    _ = cl^


def sort_op_gather_rows_f32_host(src: _HF, order: _HI, dst: _HF, rows: Int, cols: Int):
    """dst[r, c] = src[order[r], c] (core/sort_ops.mojo
    `sort_op_gather_rows_f32`), rows in parallel."""

    def row(r: Int) {imm src, imm order, imm dst, imm cols}:
        var s = Int(order.unsafe_load(r))
        for c in range(cols):
            dst.unsafe_store(r * cols + c, src.unsafe_load(s * cols + c))

    host_cells(row, rows, cols)
