# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The HOST column of `builder.IDN_ET_BINNED` (fam2-forests, 2026-10-04, a
candidate arm, default OFF): the border table and the 16-bit codes the
device builds in `DeviceDataset.ensure_binned`, restated on the host so
`train_tree_exact` searches the same binned X the device searches.

Device, in order: `compute_quantiles(ctx, d_data, ET_BINS, n_rows, n_cols)`
(default oversampling 4, seed 0, column-major: one PCG row per sample index
shared by every column, the segmented sort over twiddled keys, the Float64
bin index table, the gather, the sequential `unique` over `ftz` operands),
then `et_bin_rows_kernel` (a lower bound over `[0, n_bins - 1]` with the
compare `border >= value`). Host: the same statements, sequentially, from
the RandomForest host oracle's generator and bin index
(`ensemble/host/rf_oracle.mojo`), so a change to either lands here too.

Host only: reached from `extratrees/impl/randomforest/host_forest.mojo`.
"""

from std.builtin.sort import sort
from std.memory import bitcast

from checks.numerics import ftz
from ensemble.host.rf_oracle import (
    RF_QUANTILE_OVERSAMPLING,
    host_pcg_init,
    host_pcg_uniform_u64,
    host_quantile_bin_index,
)
from extratrees.impl.decisiontree.batched_levelalgo.builder import (
    ET_BINS,
    HostBins,
)


@always_inline
def _to_sortable(bits: UInt32) -> UInt32:
    """`float_to_sortable`, `core/segmented_sort.mojo` (CUB TwiddleIn)."""
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return ~bits
    return bits | UInt32(0x80000000)


@always_inline
def _from_sortable(key: UInt32) -> UInt32:
    """`sortable_to_float`, `core/segmented_sort.mojo` (TwiddleOut)."""
    if (key & UInt32(0x80000000)) != UInt32(0):
        return key & UInt32(0x7FFFFFFF)
    return ~key


struct HostBinTables(Movable):
    """The owner of what `HostBins` points at. Keep it alive (name it after
    the last tree) for as long as a `view()` is in use."""

    var on: Bool
    var codes: List[UInt16]
    var q: List[Float32]
    var nb: List[Int32]

    def __init__(out self):
        """The OFF tables: one element each, never read (`HostBins.on` is
        False)."""
        self.on = False
        self.codes = List[UInt16](length=1, fill=UInt16(0))
        self.q = List[Float32](length=1, fill=Float32(0.0))
        self.nb = List[Int32](length=1, fill=Int32(0))

    def view(self) -> HostBins:
        return HostBins(
            self.on,
            rebind[MutPointer[UInt16, MutUntrackedOrigin]](
                self.codes.unsafe_ptr()
            ),
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                self.q.unsafe_ptr()
            ),
            rebind[MutPointer[Int32, MutUntrackedOrigin]](
                self.nb.unsafe_ptr()
            ),
        )


def host_bin_tables(
    x_col_major: List[Float32], n_rows: Int, n_cols: Int
) raises -> HostBinTables:
    """The borders and codes of `DeviceDataset.ensure_binned`, on the host.
    `x_col_major` is the matrix the device uploads (`d_data`)."""
    if n_rows <= 0 or n_cols <= 0:
        raise Error("host_bin_tables: n_rows and n_cols must be positive")
    if len(x_col_major) < n_rows * n_cols:
        raise Error("host_bin_tables: X is shorter than n_rows * n_cols")
    var out = HostBinTables()
    var max_n_bins = ET_BINS
    var global_rows = UInt64(n_rows)
    var budget = UInt64(max_n_bins) * UInt64(RF_QUANTILE_OVERSAMPLING)
    var sample_count = Int(global_rows if global_rows < budget else budget)

    # `sample_owned_columns_kernel`: one row per sample index, shared by
    # every column; the identity when the budget covers the data.
    var sample_rows = List[Int](capacity=sample_count)
    for sample_idx in range(sample_count):
        var global_row = UInt64(sample_idx)
        if UInt64(sample_count) != global_rows:
            var gen = host_pcg_init(UInt64(0), UInt64(sample_idx), UInt64(0))
            global_row = host_pcg_uniform_u64(gen, UInt64(0), global_rows)
        sample_rows.append(Int(global_row))

    var bin_idx = List[Int](capacity=max_n_bins)
    for b in range(max_n_bins):
        bin_idx.append(host_quantile_bin_index(b, sample_count, max_n_bins))

    out.q = List[Float32](length=n_cols * max_n_bins, fill=Float32(0.0))
    out.nb = List[Int32](length=n_cols, fill=Int32(0))
    out.codes = List[UInt16](length=n_rows * n_cols, fill=UInt16(0))
    for col in range(n_cols):
        # The segmented sort: ascending over the twiddled keys (the float
        # total order; equal keys are equal bits, so stability is
        # unobservable).
        var keys = List[UInt32](capacity=sample_count)
        for s in range(sample_count):
            keys.append(
                _to_sortable(
                    bitcast[DType.uint32](
                        x_col_major[col * n_rows + sample_rows[s]]
                    )
                )
            )
        sort(keys)
        var col_q = col * max_n_bins
        # `compute_quantiles_batched_kernel`: the gather, then the
        # sequential unique over `ftz` operands (DEVIATION 403).
        for b in range(max_n_bins):
            out.q[col_q + b] = bitcast[DType.float32](
                _from_sortable(keys[bin_idx[b]])
            )
        var w = 1
        for r in range(1, max_n_bins):
            var prev = out.q[col_q + w - 1]
            var cur = out.q[col_q + r]
            if ftz(cur) != ftz(prev):
                out.q[col_q + w] = cur
                w += 1
        out.nb[col] = Int32(w)

        # `et_bin_rows_kernel`, its compare as written (`border >= value`;
        # a NaN value answers False every step and lands on the top code).
        for row in range(n_rows):
            var v = x_col_major[col * n_rows + row]
            var lo = 0
            var hi = w - 1
            while lo < hi:
                var mid = (lo + hi) // 2
                if out.q[col_q + mid] >= v:
                    hi = mid
                else:
                    lo = mid + 1
            out.codes[col * n_rows + row] = UInt16(lo)
    out.on = True
    return out^
