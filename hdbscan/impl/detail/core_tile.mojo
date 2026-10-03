# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Core distances from one tiled kernel (lane af-hdbscan2, FAST on Apple only).

`-D MOJOLEARN_HDB_CORE_TILE`. HDBSCAN reads ONE number per row from the k-NN:
the k-th smallest distance (self included). Main's route
(`reachability.mojo::compute_knn` -> `knn_self_search_resident`) copies X
twice, computes two norm vectors, runs the matrix-unit k-NN, sorts every row,
narrows the index set to Int32 and reads slot k - 1: eleven buffers and about
six waits for one float per row, and an m x k index set nobody reads.

Here `core_tile_kernel` keeps row i in registers (d <= CT_DMAX, zero-padded),
walks every row in CT_TILE-row threadgroup tiles, accumulates the squared
difference in feature order and keeps the k smallest values seen so far in a
sorted register array (one branchless swap pass per admitted candidate;
candidates not below the current k-th are rejected by one compare). The core
distance is the square root of the k-th slot. Self is a candidate (distance
0), as in the k-NN at `min_samples + 1`. A non-finite distance marks the row
NaN so `refuse_nonfinite_device` refuses it by name, as the k-NN route does.

THE VALUE. The k-th smallest of a multiset does not depend on which equal
element supplied it, so the only difference from main's FAST route is the
arithmetic of each distance: the direct `(x - y)^2` sum here (the chain the
FAST Boruvka search uses for the pair distances) against the matrix-unit
expanded form `|x|^2 + |y|^2 - 2 x.y` there. The two agree to rounding; an
exact tie that the rounding breaks differently can move a mutual
reachability weight by one ulp. IDENTICAL never takes this file.

Fits gate: the tile is CT_TILE x CT_DMAX floats (16 KB at CT_DMAX = 64),
checked against the column's threadgroup limit at compile time.
"""

from std.gpu import block_idx, thread_idx
from std.math import sqrt
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from hdbscan.impl.detail.fast_apple import HDB_CORE_TILE


comptime CT_TPB = 128
"""Rows per block, one per thread."""
comptime CT_TILE = 64
"""Reference rows staged per threadgroup tile."""
comptime CT_KMAX = 16
"""Largest k the register top-k holds; above it the k-NN route runs."""
comptime CT_DMAX = 64
"""Widest row the register copy holds; above it the k-NN route runs."""
comptime CT_SMEM_BYTES = CT_TILE * CT_DMAX * 4


def core_tile_applies(d: Int, k: Int) -> Bool:
    """Whether this build and shape take the tiled core-distance kernel."""
    comptime if not HDB_CORE_TILE:
        return False
    comptime if not lib_smem_page_fits_for[TARGET_COLUMN, CT_SMEM_BYTES]():
        return False
    return d >= 1 and d <= CT_DMAX and k >= 1 and k <= CT_KMAX


def core_tile_kernel[DMAX: Int](
    x: MutPointer[Float32, MutAnyOrigin],
    out_core: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    d_in: Int32,
    k_in: Int32,
):
    """`out_core[i]` = sqrt of the k-th smallest squared distance from row i
    to any row (itself included), or NaN when a distance is not finite."""
    var m = Int(m_in)
    var dim = Int(d_in)
    var k = Int(k_in)
    var i = Int(block_idx.x) * CT_TPB + Int(thread_idx.x)
    var live = i < m
    var xi = InlineArray[Float32, DMAX](fill=Float32(0.0))
    if live:
        comptime for t in range(DMAX):
            if t < dim:
                xi[t] = x[i * dim + t]
    var tile = stack_allocation[
        CT_TILE * DMAX, Float32, address_space = AddressSpace.SHARED
    ]()
    var top = InlineArray[Float32, CT_KMAX](fill=Float32.MAX)
    var kth = Float32.MAX
    var bad = False
    var j0 = 0
    while j0 < m:
        var e = Int(thread_idx.x)
        while e < CT_TILE * DMAX:
            var jj = j0 + e // DMAX
            var tt = e % DMAX
            if jj < m and tt < dim:
                tile[e] = x[jj * dim + tt]
            else:
                tile[e] = Float32(0.0)
            e += CT_TPB
        barrier()
        var jn = min(CT_TILE, m - j0)
        if live:
            for u in range(jn):
                var dd = Float32(0.0)
                comptime for t in range(DMAX):
                    var df = xi[t] - tile[u * DMAX + t]
                    dd += df * df
                # Exponent bits, not a float compare: FAST arithmetic may
                # assume no inf / NaN and fold the compare away.
                if (bitcast[DType.uint32](dd) & 0x7F800000) == 0x7F800000:
                    bad = True
                elif dd < kth:
                    var v = dd
                    comptime for q in range(CT_KMAX):
                        if v < top[q]:
                            var held = top[q]
                            top[q] = v
                            v = held
                    comptime for q in range(CT_KMAX):
                        if q == k - 1:
                            kth = top[q]
        barrier()
        j0 += CT_TILE
    if live:
        out_core[i] = (
            bitcast[DType.float32](UInt32(0x7FC00000)) if bad else sqrt(kth)
        )


def compute_core_dists_tile(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut core_dists: DeviceBuffer[DType.float32],
    m: Int,
    d: Int,
    k: Int,
) raises:
    """One launch: `core_dists[0:m]` from `x` (m x d row-major), k slots.
    The caller checked `core_tile_applies(d, k)`."""
    if not core_tile_applies(d, k):
        raise Error(
            "hdbscan.compute_core_dists_tile: shape d=" + String(d) + " k="
            + String(k) + " is not this kernel's; the caller gates on"
            " core_tile_applies"
        )
    var grid = (m + CT_TPB - 1) // CT_TPB if m > 0 else 1
    if d <= 16:
        ctx.enqueue_function[core_tile_kernel[16]](
            x.unsafe_ptr(), core_dists.unsafe_ptr(), Int32(m), Int32(d),
            Int32(k), grid_dim=(grid, 1, 1), block_dim=(CT_TPB, 1, 1),
        )
    elif d <= 32:
        ctx.enqueue_function[core_tile_kernel[32]](
            x.unsafe_ptr(), core_dists.unsafe_ptr(), Int32(m), Int32(d),
            Int32(k), grid_dim=(grid, 1, 1), block_dim=(CT_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[core_tile_kernel[CT_DMAX]](
            x.unsafe_ptr(), core_dists.unsafe_ptr(), Int32(m), Int32(d),
            Int32(k), grid_dim=(grid, 1, 1), block_dim=(CT_TPB, 1, 1),
        )
