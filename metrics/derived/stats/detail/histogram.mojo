# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""RAFT `cpp/include/raft/stats/detail/histogram.cuh` (ebf9268), the one
arm the metrics reach: `gmemHistKernel` (:69-84) with the `Binner`
`int(val - minLabel)` of `adjusted_rand_index.cuh:52-59`.

Also stands in for `cub::DeviceHistogram::HistogramEven` where RAFT calls
it with integer labels and unit-wide bins (`entropy.cuh:57-92`
`countLabels`, `silhouette_score.cuh:101-136` `countLabels`): CUB's
`HistogramEven` on integer data with `lower_level = min`, `upper_level =
max + 1`, `num_levels = max - min + 2` is exactly `bins[val - min] += 1`,
and CUB's output is the same integers this kernel produces. (entropy.cuh
writes them into a DOUBLE array; the conversion of an integer count to a
double is exact below 2^53 and is done on the host here, see entropy.mojo.)

Integer atomics: order-free, identity-safe, no IDENTICAL arm needed
(contingency_matrix.mojo's header). The `__match_any_sync` lane aggregation
on sm70+ is a throughput device that lands the same integers and is not
ported.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceBuffer, DeviceContext


#: `histogram.cuh` launches `ThreadsPerBlock = 256` (:86-99). SCHEDULING.
comptime HIST_TPB = 256


def gmem_hist_kernel(
    bins: MutPointer[Int32, MutAnyOrigin],
    data: MutPointer[Int32, MutAnyOrigin],
    nrows: Int32,
    min_label: Int32,
):
    """`gmemHistKernel` with `ncols = 1`, `VecLen = 1`, `Binner(minLabel)`."""
    var row = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if row < Int(nrows):
        var bin_id = Int(data.unsafe_load(row) - min_label)
        _ = Atomic.fetch_add(bins.unsafe_offset(bin_id), Int32(1))


def histogram(
    ctx: DeviceContext,
    mut bins: DeviceBuffer[DType.int32],
    nbins: Int,
    mut data: DeviceBuffer[DType.int32],
    nrows: Int,
    min_label: Int32,
) raises:
    """`raft::stats::histogram(HistTypeAuto, bins, nbins, data, nrows, 1,
    stream, Binner(minLabel))`: `bins[0:nbins]` zeroed (`:476`), then
    counted."""
    if len(bins) < nbins:
        raise Error(
            "histogram: bins holds "
            + String(len(bins))
            + " ints, needs "
            + String(nbins)
        )
    ctx.enqueue_memset(bins, Int32(0))
    var grid = ceildiv(nrows, HIST_TPB)
    if grid < 1:
        grid = 1
    ctx.enqueue_function[gmem_hist_kernel](
        bins.unsafe_ptr(),
        data.unsafe_ptr(),
        Int32(nrows),
        min_label,
        grid_dim=(grid, 1, 1),
        block_dim=(HIST_TPB, 1, 1),
    )
    ctx.synchronize()
