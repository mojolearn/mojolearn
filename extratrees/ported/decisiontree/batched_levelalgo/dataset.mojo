"""The dataset view handed to every kernel.

A PORT of cuML `cpp/src/decisiontree/batched-levelalgo/dataset.h` (43 lines),
pinned at `00094f7` in `~/CascadeProjects/upstream/cuml`. Field for field,
their names kept so the two files diff side by side.

THE ONE THING THEIR HEADER SAYS THAT IS EASY TO SKIP: `data` is COLUMN MAJOR
(`dataset.h:24`, "assumed to be col-major"). Every access below is
`data[col * M + row]`, never `row * N + col`. The range pass and the score pass
both read one feature over many rows, which is the access their layout is
chosen for.

`n_sampled_rows` and `row_ids` exist because cuML bootstraps. sklearn's
ExtraTrees defaults to `bootstrap=False` (`sklearn/ensemble/_forest.py`,
`ExtraTreesClassifier.__init__`), so for this lane `row_ids` is the identity
permutation of all `M` rows and `n_sampled_rows == M`. The field stays anyway:
it is the array the builder PARTITIONS as the frontier descends, which is its
other job in their code and is not optional.
"""


@fieldwise_init
struct Dataset(ImplicitlyCopyable, Movable):
    """`dataset.h:22-38`. Pointers are non-owning views."""

    var data: UnsafePointer[Float32, MutAnyOrigin]
    """Input dataset, COLUMN MAJOR. Theirs is `const DataT* data`."""

    var labels: UnsafePointer[Float32, MutAnyOrigin]
    """Input labels. Theirs is `const LabelT* labels`. Classification stores
    the class id as a float here, exactly as theirs does -- cuML's `LabelT` is
    the data type for regression and the class index for classification, and
    `objectives.cuh` casts it at the accumulate site."""

    var m: Int32
    """Total rows in dataset. Theirs is `IdxT M`."""

    var n: Int32
    """Total cols in dataset. Theirs is `IdxT N`."""

    var n_sampled_rows: Int32
    """Total sampled rows. Theirs is `IdxT n_sampled_rows`."""

    var n_sampled_cols: Int32
    """Total sampled cols, i.e. `max(1, max_features * N)`
    (`builder.cuh:222`). Theirs is `IdxT n_sampled_cols`."""

    var row_ids: UnsafePointer[Int32, MutAnyOrigin]
    """Indices of sampled rows. Theirs is `IdxT* row_ids`. The builder
    partitions THIS array in place as nodes split."""

    var num_outputs: Int32
    """Number of classes, or of regression outputs. Theirs is
    `IdxT num_outputs`."""

    def value(self, row: Int32, col: Int32) -> Float32:
        """One cell, honouring their column-major layout (`dataset.h:24`)."""
        return self.data[Int(col) * Int(self.m) + Int(row)]

    def label(self, row: Int32) -> Float32:
        """One label."""
        return self.labels[Int(row)]
