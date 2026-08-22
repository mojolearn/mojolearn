"""The dataset view every builder kernel reads through.

MIRRORS `cpp/src/decisiontree/batched-levelalgo/dataset.h` at
rapidsai/cuml `v26.08.00` (`265b9da6a0e75dbef071a3168398b993a5ff6f0e`),
checked out read-only at `~/CascadeProjects/upstream/cuml-v26.08.00`.

Their whole file is one struct of pointers and strides plus one accessor
(`dataset.h:40-44`):

    HDI DataT value(IdxT row, IdxT col) const
    {
      return data[static_cast<std::int64_t>(row) * row_stride +
                  static_cast<std::int64_t>(col) * col_stride];
    }

That accessor IS the memory shape of this whole learner. The histogram
kernel's inner loop (`kernels/builder_kernels_impl.cuh:336-341`) is

    for (auto i = range_start + tid; i < end; i += stride) {
      auto row   = dataset.row_ids[i];
      auto data  = dataset.value(row, col);
      auto label = dataset.labels[row];

so per sampled instance it performs TWO random-index gathers -- one into
the feature column, one into the labels -- with `row` coming from a
`row_ids` array that at the root is `raft::random::uniformInt` output
(`randomforest.cuh:140-142`), i.e. unsorted uniform draws WITH
replacement. Column-major input gives `row_stride = 1`,
`col_stride = n_rows`; row-major gives the transpose. Both are carried,
because `Builder`'s constructor takes `row_major` as a parameter
(`builder.cuh:226`).

================= DEVIATION BLOCK (whole file) =================
DEVIATION 100. `sample_weight` is `const double*` in their struct
(`dataset.h:22`) and this device has no float64. The field is carried as
Float32 here.

WHAT THIS COSTS, priced rather than waved through: `sample_weight` is
read at exactly two sites, both in `objectives.cuh`
(`:158` and `:359`), both inside `if constexpr (weighted)`, and both
immediately widen the value into a `double weight` that is then summed
into `WeightedClassificationBin::weight` / `WeightedRegressionBin::
weight`. The UNWEIGHTED instantiations -- which are what
`bootstrap=True` (their default, `randomforest.cuh:140`) selects, since
`RowSampler::tree_sample_weight()` returns `nullptr` whenever
bootstrapping is on (`randomforest.cuh:167`) -- never touch this pointer
at all. So for the path this port ships first the field is inert, and
for the weighted path it is a documented precision reduction on the
per-row weight only, not on the accumulator. The accumulator's own
float64 problem is DEVIATION 101 in `bins.mojo` and is the one that
actually bites.

DEVIATION 102. `n_rows`, `n_cols`, `row_stride`, `col_stride` are
`std::int64_t` in their struct and are Int64 here; `n_sampled_rows`,
`n_sampled_cols`, `num_outputs` and `row_ids` are `IdxT` in their struct
and IdxT is instantiated as `int` throughout cuML's RF
(`randomforest.cuh` builds `Builder<ObjectiveT>` with IdxT = int), so
they are Int32 here. That is not a deviation in value, only in spelling
-- it is recorded so that a reader who finds an Int32 beside an Int64 in
this file knows the widths came from their header and not from a guess.
Kernel-parameter scalars are Int32 per this repository's rule (a Metal
kernel argument must be Int32), and the two Int64 strides are therefore
passed as a `DatasetView`, not as loose scalars.
=================================================================
"""


@fieldwise_init
struct DatasetView[dtype: DType, label_dtype: DType](Copyable, Movable):
    """`ML::DT::Dataset<DataT, LabelT, IdxT>`, `dataset.h:15-45`.

    Every field keeps their name so the two files diff side by side.
    """

    # `dataset.h:18` -- input dataset
    var data: MutPointer[Scalar[Self.dtype], MutUntrackedOrigin]
    # `dataset.h:20` -- input labels
    var labels: MutPointer[Scalar[Self.label_dtype], MutUntrackedOrigin]
    # `dataset.h:22` -- optional input sample weights; see DEVIATION 100
    var sample_weight: MutPointer[Float32, MutUntrackedOrigin]
    # `dataset.h:24` -- total rows in dataset
    var n_rows: Int64
    # `dataset.h:26` -- total cols in dataset
    var n_cols: Int64
    # `dataset.h:28` -- row stride in input data elements
    var row_stride: Int64
    # `dataset.h:30` -- column stride in input data elements
    var col_stride: Int64
    # `dataset.h:32` -- total sampled rows in dataset
    var n_sampled_rows: Int32
    # `dataset.h:34` -- total sampled cols in dataset
    var n_sampled_cols: Int32
    # `dataset.h:36` -- indices of sampled rows
    var row_ids: MutPointer[Int32, MutUntrackedOrigin]
    # `dataset.h:38` -- number of classes or regression outputs
    var num_outputs: Int32
    # NOT in their struct: their `sample_weight == nullptr` test
    # (`objectives.cuh:158`) has no Mojo counterpart, because
    # `MutPointer` is non-null by design. The null-ness their code tests
    # is carried as this flag and tested in the same two places.
    var has_sample_weight: Bool
    # DEVIATION 314 -- NOT in their struct: the pre-binned dataset. Their
    # histogram kernel re-derives `lower_bound(quantiles[col], value)`
    # for every element at EVERY level of EVERY tree
    # (`builder_kernels_impl.cuh:341`), but that index is a pure function
    # of (row, col) once the per-forest quantiles exist
    # (`randomforest.cuh:317-325` computes them ONCE). `bins` holds that
    # index, computed once per forest by `launch_bin_dataset`
    # (quantiles.mojo), laid out with the SAME strides as `data`, one
    # uint8 per element -- so the histogram kernel's 4-byte random gather
    # plus 7-step binary search becomes a 1-byte read of the SAME index,
    # and every downstream bit is unchanged. Launch-log attribution
    # measured that kernel at 85.3% of device time at 500k x 50
    # (bench/results/RF_2026-08-22_attribution.md). `has_bins` is False
    # when `max_n_bins > 256` (uint8 cannot hold the index) and in
    # checks that exercise the raw path; the kernel then searches as
    # theirs does.
    var bins: MutPointer[UInt8, MutUntrackedOrigin]
    var has_bins: Bool

    @always_inline
    def value(self, row: Int32, col: Int32) -> Scalar[Self.dtype]:
        """`dataset.h:40-44`. The int64 casts are theirs, not padding."""
        return self.data[
            unsafe_offset = Int(
                Int64(Int(row)) * self.row_stride
                + Int64(Int(col)) * self.col_stride
            )
        ]

    @always_inline
    def bin_of(self, row: Int32, col: Int32) -> Int32:
        """DEVIATION 314: the precomputed `lower_bound` index for
        (row, col). Valid only when `has_bins`; same offset formula as
        `value` above."""
        return Int32(
            Int(
                self.bins[
                    unsafe_offset = Int(
                        Int64(Int(row)) * self.row_stride
                        + Int64(Int(col)) * self.col_stride
                    )
                ]
            )
        )
