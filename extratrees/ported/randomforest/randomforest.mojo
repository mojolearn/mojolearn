"""The forest: many ExtraTrees, and the vote that turns them into a model.

A PORT of cuML `cpp/src/randomforest/randomforest.cuh`, pinned at `00094f7`:

| ours | theirs |
|---|---|
| `row_sample_for` | `randomforest.cuh:50-72` (`get_row_sample`) |
| `fit_classification` / `fit_regression` | `:155-195` (the tree loop) |
| `predict_class_forest` / `predict_regression_forest` | `:223-256` |
| `error_checking` | `:74-90` |

**THE TREES ARE NOT COPIES OF EACH OTHER, AND THAT IS THE KEY.** Their
`get_row_sample` (`:59-62`) hashes `(seed, tree_id)` with the fnv1a32 chain
before seeding the row sampler; our tree ids enter the split key the same way
(`key_for(seed, tree_id, node_id, feature_id)`, deviation 130), so tree `i` and
tree `j` draw different thresholds on identical data. `tree_check` asserts
exactly that, because a forest whose trees are identical is one tree reported a
hundred times.

**`bootstrap=False` IS sklearn's ExtraTrees default and it simplifies this
file to almost nothing.** cuML's `get_row_sample` (`:68-70`) takes the
`thrust::sequence` branch when bootstrap is off — the identity permutation,
every row, every tree. So there is no row sampler here at all. What there IS,
and what is easy to miss: **each tree needs its OWN `row_ids` buffer**, because
`train_*` PARTITIONS it in place. Theirs is per-stream
(`selected_rows[stream_id]`, `:169`); ours is per tree.

**THE PREDICTION IS AN ACCUMULATION, AND IT IS WHY DEVIATION 147 EXISTS.**
Their `predict` (`:229-242`) declares `row_prediction(num_outputs)` INSIDE the
row loop — value-initialised to zero — calls the per-tree predictor once per
tree into that same buffer with `+=`, and only then divides by `n_trees`. The
accumulation across trees IS the forest. That is why `flatnode.mojo` ports the
accumulating form under a name that says so; this file is its only caller so
far, and it must not reach for the zeroing convenience wrapper.
"""

from extratrees.ported.decisiontree.decisiontree import (
    DecisionTreeParams,
    validity_check,
)
from extratrees.ported.decisiontree.flatnode import (
    TreeMetaDataNode,
    predict_one_accumulate,
)
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    train_classification,
    train_regression,
)
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset


comptime BOOTSTRAP_DEFAULT = False
"""sklearn's `ExtraTreesClassifier`/`Regressor` default. cuML's `RF_params`
defaults it TRUE (`randomforest.hpp`), which is Random Forest's default, not
ExtraTrees'. This directory is ExtraTrees, so the default is sklearn's, and
`fit_*` REFUSES `bootstrap=True` rather than silently ignoring it — the row
sampler is not ported."""


def row_sample_for(n_rows: Int32, bootstrap: Bool) raises -> List[Int32]:
    """`randomforest.cuh:50-72`, the `bootstrap == false` arm.

    Theirs is `thrust::sequence(...)` over `selected_rows` (`:69`) — the
    identity permutation. The bootstrap arm (`:66`) draws
    `uniformInt(0, n_rows)` with replacement through a Philox generator seeded
    by `fnv1a32(fnv1a32(basis, seed), tree_id)`; it is NOT ported, because
    sklearn's ExtraTrees does not use it and porting a sampler nothing calls
    would be an unwired file (rule 3).
    """
    if bootstrap:
        raise Error(
            "bootstrap=True is not ported in extratrees/: sklearn's"
            " ExtraTrees defaults to bootstrap=False and the row sampler"
            " (randomforest.cuh:66) has no caller here. Refused by name"
            " rather than silently ignored."
        )
    var out = List[Int32]()
    for r in range(Int(n_rows)):
        out.append(Int32(r))
    return out^


struct Forest(Movable):
    """`RandomForestMetaData` reduced to what this lane fills.

    Theirs carries `trees`, `rf_params` and a `n_streams` worth of scratch.
    Ours carries the trees and the two numbers a prediction needs.
    """

    var trees: List[TreeMetaDataNode[DType.float32]]
    var num_outputs: Int32
    var n_trees: Int32

    def __init__(out self, num_outputs: Int32):
        self.trees = List[TreeMetaDataNode[DType.float32]]()
        self.num_outputs = num_outputs
        self.n_trees = 0


def error_checking(n_rows: Int32, n_cols: Int32, n_trees: Int32) raises:
    """`randomforest.cuh:74-90`, the fit arm."""
    if n_rows <= 0:
        raise Error("Invalid n_rows " + String(n_rows))
    if n_cols <= 0:
        raise Error("Invalid n_cols " + String(n_cols))
    if n_trees <= 0:
        raise Error("Invalid n_trees " + String(n_trees))


def fit_classification(
    x_col_major: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_cols: Int32,
    n_classes: Int32,
    params: DecisionTreeParams,
    n_trees: Int32,
    seed: UInt64,
    bootstrap: Bool = BOOTSTRAP_DEFAULT,
) raises -> Forest:
    """`randomforest.cuh:155-195`, the tree loop.

    Theirs runs the loop under OpenMP across `n_streams` CUDA streams
    (`:161-167`). Ours is serial, and that is not a deviation to record but a
    consequence of a fact already in the traps register: **Metal has no
    streams**, so their overlap has nothing to port onto. The trees are
    independent either way — tree `i` reads `x` and writes its own `row_ids`
    and its own tree — so the answer does not depend on the order they run in,
    which is the property that makes the serial form a faithful stand-in
    rather than a different algorithm.
    """
    error_checking(n_rows, n_cols, n_trees)
    validity_check(params)
    if n_classes < 1:
        raise Error("n_classes must be >= 1; got " + String(n_classes))

    var forest = Forest(n_classes)
    for tree_id in range(Int(n_trees)):
        # `:169` -- each tree gets its OWN row list, because `train_*`
        # partitions it in place.
        var row_ids = row_sample_for(n_rows, bootstrap)
        var dataset = Dataset(
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                x_col_major.unsafe_ptr()
            ),
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                labels.unsafe_ptr()
            ),
            n_rows,
            n_cols,
            n_rows,
            n_cols,
            rebind[MutPointer[Int32, MutUntrackedOrigin]](
                row_ids.unsafe_ptr()
            ),
            n_classes,
        )
        # `:180-191` -- `i` is passed as the tree id, which is what makes the
        # trees differ (`:59-62` hashes it into the seed).
        forest.trees.append(
            train_classification(
                dataset, params, Int32(tree_id), seed, n_classes
            )
        )
        _ = row_ids.unsafe_ptr()
    forest.n_trees = n_trees
    return forest^


def fit_regression(
    x_col_major: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_cols: Int32,
    params: DecisionTreeParams,
    n_trees: Int32,
    seed: UInt64,
    bootstrap: Bool = BOOTSTRAP_DEFAULT,
) raises -> Forest:
    """The regression arm of the same loop."""
    error_checking(n_rows, n_cols, n_trees)
    validity_check(params)

    var forest = Forest(1)
    for tree_id in range(Int(n_trees)):
        var row_ids = row_sample_for(n_rows, bootstrap)
        var dataset = Dataset(
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                x_col_major.unsafe_ptr()
            ),
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                labels.unsafe_ptr()
            ),
            n_rows,
            n_cols,
            n_rows,
            n_cols,
            rebind[MutPointer[Int32, MutUntrackedOrigin]](
                row_ids.unsafe_ptr()
            ),
            1,
        )
        forest.trees.append(
            train_regression(dataset, params, Int32(tree_id), seed)
        )
        _ = row_ids.unsafe_ptr()
    forest.n_trees = n_trees
    return forest^


def forest_vote(
    forest: Forest, row: List[Float32], row_offset: Int
) raises -> List[Float32]:
    """The averaged per-class (or per-output) prediction for one row.

    `randomforest.cuh:229-242`, transcribed:

        std::vector<T> row_prediction(num_outputs);   // zero-initialised
        for (i in trees) predict(..., row_prediction.data(), ...);   // +=
        for (k) row_prediction[k] /= n_trees;

    The zero-initialisation is `std::vector`'s constructor and the `+=` is in
    `predict_one` (`decisiontree.cuh:411`). Both halves matter: this is the
    ONLY place the accumulating form of `predict_one` is called, and reaching
    for `flatnode.mojo`'s zeroing wrapper here would silently return the last
    tree's leaf instead of the forest's average. Deviation 147.
    """
    if forest.n_trees < 1:
        raise Error("an empty forest cannot predict")
    var acc = List[Float32](length=Int(forest.num_outputs), fill=Float32(0.0))
    for i in range(len(forest.trees)):
        predict_one_accumulate(
            row,
            row_offset,
            forest.trees[i],
            acc,
            0,
            Int(forest.num_outputs),
        )
    for k in range(Int(forest.num_outputs)):
        acc[k] = acc[k] / Float32(Int(forest.n_trees))
    return acc^


def predict_class_forest(
    forest: Forest, row: List[Float32], row_offset: Int
) raises -> Int:
    """`randomforest.cuh:243-253`, the majority vote, transcribed including
    both of its quirks: `best_prob` starts at `0.0` rather than `-inf`, and the
    comparison is strictly greater while `k` ascends — so an exact tie keeps
    the LOWEST class index, and a row whose averaged scores are all `<= 0`
    returns class 0 without the comparison ever firing."""
    var acc = forest_vote(forest, row, row_offset)
    var best_class = 0
    var best_prob = Float32(0.0)
    for k in range(Int(forest.num_outputs)):
        if acc[k] > best_prob:
            best_class = k
            best_prob = acc[k]
    return best_class


def predict_regression_forest(
    forest: Forest, row: List[Float32], row_offset: Int
) raises -> Float32:
    """`randomforest.cuh:254-256`: `h_predictions[row_id] = row_prediction[0]`
    after the division by `n_trees`."""
    var acc = forest_vote(forest, row, row_offset)
    return acc[0]
