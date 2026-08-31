# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
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

**`bootstrap=False` IS sklearn's ExtraTrees default.** cuML's
`get_row_sample` (`:68-70`) takes the `thrust::sequence` branch when bootstrap
is off — the identity permutation, every row, every tree. **`bootstrap=True`
is honoured too, since DEVIATION 460** (2026-08-23): the `:64-67` arm draws
`n_sampled_rows` rows with replacement through RAFT's Philox `uniformInt`,
seeded by the `:59-62` fnv1a32 chain over `(seed, tree_id)` — the host form is
`core.philox.uniform_int_host`, the device form `launch_uniform_int`, BOTH the
RF lane's ports (`ensemble/`), reused rather than re-invented, with the seed
chain in `mojo_only/pcg_rng.mojo::row_sample_seed`. `n_sampled_rows` is
sklearn's `max_samples` resolved to a count (None = `n_rows`). What is easy to
miss either way: **each tree needs its OWN `row_ids` buffer**, because
`train_*` PARTITIONS it in place. Theirs is per-stream
(`selected_rows[stream_id]`, `:169`); ours is per tree.

**THE PREDICTION IS AN ACCUMULATION, AND IT IS WHY DEVIATION 147 EXISTS.**
Their `predict` (`:229-242`) declares `row_prediction(num_outputs)` INSIDE the
row loop — value-initialised to zero — calls the per-tree predictor once per
tree into that same buffer with `+=`, and only then divides by `n_trees`. The
accumulation across trees IS the forest. That is why `flatnode.mojo` ports the
accumulating form under a name that says so; this file is its only caller so
far, and it must not reach for the zeroing convenience wrapper.

**THE FOREST HAS A DEVICE ARM, AND IT IS THE SAME LOOP.** `fit_classification`
calls the host trainer per tree; `fit_classification_device` calls
`train_classification_device` per tree instead, and nothing else about the loop
changes — same `error_checking`, same `row_sample_for` seed per tree, same
per-tree `row_ids`, same `i` passed as the tree id. That is the whole of the
difference, and it is why the two produce the SAME forest: deviation 183 closed
the last gap between the device and host trees, so tree `i` of the device
forest is bit-identical to tree `i` of the host forest in `(colid, quesval,
left_child_id, instance_count)` and in every leaf value.

Three things about the device arm are stated here because they are decisions,
not consequences:

* **the dataset is uploaded once FOR THE FOREST** — DEVIATION 184, closed.
  It was once per tree for one round, because `train_classification_device`
  owned its own upload prologue; that function is now split into
  `upload_dataset` and `train_classification_device_resident` and this loop
  hoists the upload above it;
* **`row_ids` is per-tree even though the device path cannot corrupt it** —
  the device trainer partitions on the DEVICE and never copies the permutation
  back, so a shared buffer would currently be harmless. It is kept per-tree
  anyway and the harmlessness is PINNED by a check rather than assumed.
  DEVIATION 185;
* **the labels are cast to `Int32` class ids once for the forest**, not once
  per tree, because the cast does not depend on the tree id. DEVIATION 186.

`fit_regression_device` is the regression twin of the same loop, added when
deviation 206 brought the regression device path level: same upload-once
dataset (DEVIATION 184's shape), same per-tree `row_ids`, same `bootstrap`
handling, with the workspace also hoisted to the fit because deviation
202 gave regression a per-tree workspace to hoist. It takes labels ALREADY
QUANTIZED (deviation 135) plus the scale that produced them; the estimator's
`fit_extra_trees_regressor_device` derives that quantization so callers do not
have to. DEVIATION 188, which refused a device regressor by name while
`train_regression_device` did not exist, is CLOSED.
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
    train_classification_device,
    train_forest_classification_device,
    train_forest_regression_device,
    train_regression,
    upload_dataset,
)
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.mojo_only.pcg_rng import row_sample_seed
from core.philox import RNG_STRIDE, uniform_int_host
from max.gpu.host import DeviceContext


comptime BOOTSTRAP_DEFAULT = False
"""sklearn's `ExtraTreesClassifier`/`Regressor` default. cuML's `RF_params`
defaults it TRUE (`randomforest.hpp`), which is Random Forest's default, not
ExtraTrees'. This directory is ExtraTrees, so the default is sklearn's;
`bootstrap=True` is honoured (DEVIATION 460), never silently ignored."""


def resolve_n_sampled_rows(
    n_rows: Int32, bootstrap: Bool, n_sampled_rows: Int32
) raises -> Int32:
    """`selected_rows.size()`: `n_sampled_rows` when bootstrapping (0 = all
    `n_rows`, sklearn's `max_samples=None`), exactly `n_rows` otherwise --
    the identity permutation has no other width (`:69`), so a caller who
    sets a count without bootstrap is refused by name, as sklearn refuses
    `max_samples` without `bootstrap` (`_forest.py:411-416`)."""
    if bootstrap:
        if n_sampled_rows < 0:
            raise Error(
                "n_sampled_rows must be >= 0; got " + String(n_sampled_rows)
            )
        return n_rows if n_sampled_rows == 0 else n_sampled_rows
    if n_sampled_rows != 0 and n_sampled_rows != n_rows:
        raise Error(
            "n_sampled_rows="
            + String(n_sampled_rows)
            + " without bootstrap: the identity permutation (randomforest"
            ".cuh:69) is n_rows wide. sklearn refuses max_samples without"
            " bootstrap=True for the same reason (_forest.py:411)."
        )
    return n_rows


def row_sample_for(
    n_rows: Int32,
    bootstrap: Bool,
    n_sampled_rows: Int32 = 0,
    seed: UInt64 = 0,
    tree_id: Int32 = 0,
) raises -> List[Int32]:
    """`randomforest.cuh:50-72`, `get_row_sample`, BOTH arms, on the host.

    `bootstrap == false` (`:68-70`): `thrust::sequence(...)` over
    `selected_rows` -- the identity permutation.

    `bootstrap == true` (`:64-67`, DEVIATION 460):

        rng.uniformInt<int>(selected_rows->data(), selected_rows->size(),
                            0, n_rows, stream);

    under `raft::random::Rng rng(rs, GenPhilox)` with `rs` from `:59-62`
    (`row_sample_seed`). The host form is `core.philox.uniform_int_host`
    with the RF lane's pinned stride (`RNG_STRIDE`, its DEVIATION 184: the
    4 x 108 x 256 geometry of a 108-SM part, because RAFT's mapping is a
    function of the GPU model and ONE constant has to be chosen), so the
    list here is BIT-IDENTICAL to the device slot `launch_uniform_int`
    fills in `builder.mojo::fill_row_slots` -- that is what makes the host
    and device arms train the same bootstrap forest, and `forest_check`
    holds them to it.
    """
    var n = resolve_n_sampled_rows(n_rows, bootstrap, n_sampled_rows)
    if bootstrap:
        return uniform_int_host(
            UInt64(Int(row_sample_seed(seed, tree_id))),
            UInt64(0),
            RNG_STRIDE,
            Int(n),
            Int32(0),
            n_rows,
        )
    var out = List[Int32]()
    for r in range(Int(n)):
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
    n_sampled_rows: Int32 = 0,
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

    var n_sampled = resolve_n_sampled_rows(n_rows, bootstrap, n_sampled_rows)
    var forest = Forest(n_classes)
    for tree_id in range(Int(n_trees)):
        # `:169` -- each tree gets its OWN row list, because `train_*`
        # partitions it in place. `:59-67` -- the bootstrap arm is keyed by
        # `(seed, tree_id)` (DEVIATION 460).
        var row_ids = row_sample_for(
            n_rows, bootstrap, n_sampled, seed, Int32(tree_id)
        )
        var dataset = Dataset(
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                x_col_major.unsafe_ptr()
            ),
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                labels.unsafe_ptr()
            ),
            n_rows,
            n_cols,
            n_sampled,
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


def class_ids_for(
    labels: List[Float32], n_rows: Int32, n_classes: Int32
) raises -> List[Int32]:
    """The labels as the `Int32` class ids the device trainer takes.

    DEVIATION BLOCK — DEVIATION 186. THE CAST IS HOISTED OUT OF THE TREE LOOP.

    `train_classification_device` takes `class_ids: List[Int32]` because
    DEVIATION 174 rules that the device sees only integers and performs no
    float-to-int conversion of its own. The host trainer takes the same labels
    as `Float32` and truncates them where it needs an index —
    `Int(dataset.labels[unsafe_offset=row])`,
    `builder.mojo:372` — so `Int(labels[r])` here is the SAME truncation, not
    a second convention.

    It is done ONCE for the whole forest rather than once per tree. The cast
    depends on nothing that varies across trees, so hoisting it cannot change
    an answer; that is the only reason it is allowed to be hoisted while the
    dataset upload (DEVIATION 184) is not.

    **THE RANGE CHECK IS NEW AND IT IS NOT ON THE HOST PATH.** A class id
    outside `[0, n_classes)` indexes the score kernel's per-cell accumulator,
    which is a `MAX_ACC`-wide shared-memory array — an out-of-range label is an
    out-of-bounds shared write, not a wrong answer. The host path indexes host
    `List`s instead and is refused elsewhere or crashes loudly. This is a guard
    on a cast this file owns, not a change to a ported file, and it is stated
    rather than left to be discovered because it means the two arms can refuse
    DIFFERENT inputs: a label of `7.0` with `n_classes == 3` reaches the host
    trainer and is refused here.
    """
    if len(labels) != Int(n_rows):
        raise Error(
            "labels must be n_rows long; got "
            + String(len(labels))
            + " for n_rows="
            + String(n_rows)
        )
    var out = List[Int32]()
    for r in range(Int(n_rows)):
        var c = Int(labels[r])
        if c < 0 or c >= Int(n_classes):
            raise Error(
                "label "
                + String(labels[r])
                + " at row "
                + String(r)
                + " truncates to class id "
                + String(c)
                + ", which is outside [0, "
                + String(n_classes)
                + "). The device score kernel would index its shared"
                " accumulator out of bounds; refused by name."
            )
        out.append(Int32(c))
    return out^


def fit_classification_device(
    ctx: DeviceContext,
    x_col_major: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_cols: Int32,
    n_classes: Int32,
    params: DecisionTreeParams,
    n_trees: Int32,
    seed: UInt64,
    bootstrap: Bool = BOOTSTRAP_DEFAULT,
    n_sampled_rows: Int32 = 0,
) raises -> Forest:
    """`randomforest.cuh:155-195` again, with the split search on the GPU.

    Everything the host loop does, this does: `error_checking` and
    `validity_check` first, the bootstrap count resolved through
    `resolve_n_sampled_rows` (the draw itself is the device's,
    `fill_row_slots`, DEVIATION 460), and `i` passed as the tree id — which is what makes the
    trees differ (deviation 130 hashes it into the split key). The argument
    list is the host one plus a `DeviceContext`, and the labels stay
    `Float32` on purpose so a caller can hand the SAME data to either arm and
    compare.

    What is NOT the host loop any more: there is no per-tree loop. DEVIATION
    184 (CLOSED) uploads the dataset once for the forest, and DEVIATION 211
    hands the whole tree-id list to `train_forest_classification_device`,
    which merges every tree's frontier into shared level batches — cuML's
    stream-pool overlap (`randomforest.cuh:336-341`) expressed as a wider
    grid, because Metal has no streams. `device_forest_check` still holds
    this forest to the HOST forest node for node, and `device_batched_check`
    holds it to one-tree device builds, so neither move can have changed a
    tree.
    """
    error_checking(n_rows, n_cols, n_trees)
    validity_check(params)
    if n_classes < 1:
        raise Error("n_classes must be >= 1; got " + String(n_classes))
    if len(x_col_major) != Int(n_rows) * Int(n_cols):
        raise Error(
            "x_col_major must be n_rows * n_cols long, column major; got "
            + String(len(x_col_major))
        )

    # DEVIATION 186: once for the forest, not once per tree.
    var class_ids = class_ids_for(labels, n_rows, n_classes)

    # DEVIATION 184, CLOSED: the dataset is uploaded ONCE FOR THE FOREST and
    # every tree reads the resident copy, which is what cuML's `Dataset` does
    # (`dataset.h:22-38`). This used to be an upload per tree -- `n_trees - 1`
    # redundant copies of the same immutable matrix -- because
    # `train_classification_device` owned its own prologue. That function is
    # now split into `upload_dataset` and
    # `train_classification_device_resident`, with the old name kept as a
    # wrapper for single-tree callers.
    var device_dataset = upload_dataset(
        ctx, x_col_major, class_ids, n_rows, n_cols, n_classes
    )

    # The count is resolved (and a bad one refused BY NAME) once for the
    # forest; the rows themselves are drawn ON THE DEVICE -- deviation 200
    # fills the identity with a sequence KERNEL, DEVIATION 460 fills a
    # bootstrap slot with `launch_uniform_int`.
    var n_sampled = resolve_n_sampled_rows(n_rows, bootstrap, n_sampled_rows)

    # DEVIATION 211: the whole forest through ONE merged-frontier trainer --
    # every tree's level runs in the same launches -- instead of a per-tree
    # loop of `train_classification_device_resident` calls. `:180-191`'s
    # "i is passed as the tree id" survives as the `tree_ids` list, and the
    # trees are the SAME TREES: `device_batched_check` holds the merged
    # forest to the one-tree builds node for node.
    #
    # Deviation 185's old block here (the vacuous per-tree `mut row_ids`,
    # measured, and the warning that a device-resident frontier would make a
    # shared HOST buffer dangerous) is superseded by exactly that frontier
    # arriving: each in-flight tree owns a SLOT of one device buffer, which
    # is the isolation that block said would be needed, and the check that
    # pinned the old fact now gates the slots (FOREST_SAB_SHARED_ROW_BASE).
    var tree_ids = List[Int32]()
    for tree_id in range(Int(n_trees)):
        tree_ids.append(Int32(tree_id))
    var forest = Forest(n_classes)
    forest.trees = train_forest_classification_device(
        ctx, device_dataset, params, tree_ids, seed,
        bootstrap=bootstrap, n_sampled_rows=n_sampled,
    )
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
    n_sampled_rows: Int32 = 0,
) raises -> Forest:
    """The regression arm of the same loop."""
    error_checking(n_rows, n_cols, n_trees)
    validity_check(params)

    var n_sampled = resolve_n_sampled_rows(n_rows, bootstrap, n_sampled_rows)
    var forest = Forest(1)
    for tree_id in range(Int(n_trees)):
        var row_ids = row_sample_for(
            n_rows, bootstrap, n_sampled, seed, Int32(tree_id)
        )
        var dataset = Dataset(
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                x_col_major.unsafe_ptr()
            ),
            rebind[MutPointer[Float32, MutUntrackedOrigin]](
                labels.unsafe_ptr()
            ),
            n_rows,
            n_cols,
            n_sampled,
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


def fit_regression_device(
    ctx: DeviceContext,
    x_col_major: List[Float32],
    labels_q: List[Int32],
    scale: Float64,
    n_rows: Int32,
    n_cols: Int32,
    params: DecisionTreeParams,
    n_trees: Int32,
    seed: UInt64,
    bootstrap: Bool = BOOTSTRAP_DEFAULT,
    n_sampled_rows: Int32 = 0,
) raises -> Forest:
    """A regression forest with its split search on the GPU.

    `fit_classification_device`'s twin. The dataset is immutable across
    trees, so it is uploaded ONCE (DEVIATION 184 -- going through
    `train_regression_device` per tree instead re-uploads it `n_trees` times
    and rebuilds the workspace each time, MEASURED at roughly 100 ms per tree
    of pure floor at 100,000 rows), and the whole forest runs through
    `train_forest_regression_device`'s merged frontier (DEVIATION 211), which
    owns the workspace -- one per group now, deviation 202 taken further.

    `labels_q` is already quantized (DEVIATION 135); `scale` puts the leaf
    values back in the label's units.
    """
    error_checking(n_rows, n_cols, n_trees)
    validity_check(params)
    var dataset = upload_dataset(
        ctx, x_col_major, labels_q, n_rows, n_cols, 1
    )
    # The count is resolved once for the forest; the rows are drawn on the
    # device (deviation 200's sequence kernel, DEVIATION 460's Philox draw).
    var n_sampled = resolve_n_sampled_rows(n_rows, bootstrap, n_sampled_rows)
    # DEVIATION 211: one merged-frontier trainer for the whole forest; the
    # per-group workspace lives inside it now (deviation 202, further).
    var tree_ids = List[Int32]()
    for tree_id in range(Int(n_trees)):
        tree_ids.append(Int32(tree_id))
    var forest = Forest(1)
    forest.trees = train_forest_regression_device(
        ctx, dataset, scale, params, tree_ids, seed,
        bootstrap=bootstrap, n_sampled_rows=n_sampled,
    )
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
