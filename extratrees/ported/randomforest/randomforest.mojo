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

**THE FOREST HAS A DEVICE ARM, AND IT IS THE SAME LOOP.** `fit_classification`
calls the host trainer per tree; `fit_classification_device` calls
`train_classification_device` per tree instead, and nothing else about the loop
changes — same `error_checking`, same `bootstrap=True` refusal by name, same
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
refusal by name, with the workspace also hoisted to the fit because deviation
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
    make_level_workspace,
    train_regression_device_resident,
    n_sampled_cols_for,
    DEVICE_TPB,
    train_classification,
    train_classification_device,
    train_classification_device_resident,
    train_regression,
    upload_dataset,
)
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from max.gpu.host import DeviceContext


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
) raises -> Forest:
    """`randomforest.cuh:155-195` again, with the split search on the GPU.

    Line for line `fit_classification` above, with `train_classification`
    replaced by `train_classification_device`. Everything the host loop does,
    this does: `error_checking` and `validity_check` first, `bootstrap=True`
    REFUSED BY NAME through `row_sample_for`, a FRESH `row_ids` per tree, and
    `i` passed as the tree id — which is what makes the trees differ
    (deviation 130 hashes it into the split key). The argument list is the host
    one plus a `DeviceContext`, and the labels stay `Float32` on purpose so a
    caller can hand the SAME data to either arm and compare.

    DEVIATION BLOCK — DEVIATION 184. THE DATASET IS UPLOADED ONCE PER TREE.

    **Theirs.** cuML's `Dataset` holds device pointers for the whole fit
    (`dataset.h:22-38`); `fit` uploads `X` and `y` once (`randomforest.cuh:158`
    region) and every tree in the loop reads the same resident copy.

    **Ours.** `train_classification_device` allocates `d_data` and `d_labels`
    and fills them from host staging buffers on entry
    (`builder.mojo:871-889`), because it was written as a whole-tree entry
    point with no forest above it. So an `n_trees`-tree forest performs
    `n_trees` uploads of the same immutable `n_rows * n_cols` matrix and
    `n_trees` uploads of the same label vector.

    **Why it is left that way IN THIS ROUND, stated as a decision.** Hoisting
    the upload means `train_classification_device` must accept device buffers
    instead of `List`s, which changes its signature — and `builder.mojo` is
    owned by another session this round. Changing a file two sessions are
    editing is the failure mode rule 12 names (file convergence, not
    delegation, is what predicts integration pain). The alternative — writing
    a second, forest-private copy of that 500-line function with a different
    prologue — would put the device tree build in two places and guarantee
    they drift.

    **THE PRICE, WRITTEN DOWN AND NOT MEASURED IN TIME** (no duration is taken
    anywhere in this lane): `n_trees - 1` redundant host-to-device copies of
    `4 * n_rows * n_cols` bytes plus `4 * n_rows` bytes, and the same number of
    redundant host-side staging fills and `synchronize()` points. It is
    redundant TRAFFIC and redundant SYNCHRONISATION; it is not a wrong answer,
    and it cannot become one, because the matrix is immutable and every tree
    uploads the identical bytes. That is exactly why it was safe to defer: the
    cost is a number nobody is allowed to take yet, and the correctness is
    unaffected.

    **WHAT THE NEXT ROUND SHOULD DO, precisely.** In `builder.mojo`, split
    `train_classification_device` at line 889 (the end of its upload prologue)
    into two functions:

      1. a `DeviceDataset` struct holding `d_data`, `d_labels` and `n_rows`,
         `n_cols`, `n_classes`, built by a `upload_dataset(ctx, x_col_major,
         class_ids, n_rows, n_cols)` that is the current lines 871-889;
      2. `train_classification_device_resident(ctx, dataset, mut row_ids, ...)`
         — the current body from line 891 on, taking `d_data`/`d_labels` from
         the struct and allocating only `d_row_ids` per tree, since `row_ids`
         is the one input that is per-tree.

    Keep the present `train_classification_device` as a two-line wrapper
    (upload, then call the resident form) so `device_tree_check` and every
    other existing caller is untouched. Then this function hoists
    `upload_dataset` above the tree loop and passes the same struct to every
    tree. The identity claim is unchanged by that move and
    `device_forest_check` is the check that proves it: it compares the device
    forest against the host forest node for node, so a hoisted upload that
    corrupted anything turns it red.
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

    var forest = Forest(n_classes)
    for tree_id in range(Int(n_trees)):
        # `:169` -- each tree gets its OWN row list. On the HOST path that is
        # load-bearing because `train_*` partitions the list in place; on this
        # path the partition happens on the DEVICE, in `d_row_ids`, and the
        # host list is never written back.
        #
        # DEVIATION BLOCK -- DEVIATION 185. THE PER-TREE `row_ids` IS A
        # CONTRACT HERE, NOT A NECESSITY, AND THAT IS MEASURED.
        #
        # `train_classification_device` declares `mut row_ids` and reads it
        # once, into a host staging buffer, before uploading
        # (`builder.mojo:884-885`). `node_split_kernel` then mutates
        # `d_row_ids` on the device across every level, and nothing ever copies
        # that permutation back. So the `mut` is currently vacuous and a single
        # shared buffer across all trees would produce a BIT-IDENTICAL forest.
        # `device_forest_check` MEASURES that -- it fits the whole forest from
        # one shared buffer and finds 0 differing nodes -- rather than leaving
        # it as an argument.
        #
        # It is kept per-tree regardless, for two reasons that are not style:
        # (a) the two arms of this file must be the same loop, and the host arm
        # NEEDS it; (b) the moment the device partition is copied back -- which
        # is what a device-resident frontier would do -- a shared buffer
        # becomes silent cross-tree corruption, and the tree that noticed would
        # be tree 1. The check therefore PINS the current fact: it asserts that
        # `row_ids` comes back from the device trainer unchanged. When that
        # assertion goes red, this comment is stale and the shared buffer has
        # become dangerous.
        var row_ids = row_sample_for(n_rows, bootstrap)
        # `:180-191` -- `i` is passed as the tree id.
        forest.trees.append(
            train_classification_device_resident(
                ctx,
                device_dataset,
                row_ids,
                params,
                Int32(tree_id),
                seed,
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
) raises -> Forest:
    """A regression forest with its split search on the GPU.

    `fit_classification_device`'s twin, and it exists for DEVIATION 184's
    reason: the dataset is immutable across trees, so it is uploaded ONCE, and
    the level workspace is allocated once for the same reason DEVIATION 202
    allocates it once per tree rather than once per level. Going through
    `train_regression_device` per tree instead uploads the same
    `4*n_rows*n_cols + 4*n_rows` bytes `n_trees` times and rebuilds the
    workspace `n_trees` times -- MEASURED at roughly 100 ms per tree of pure
    floor at 100,000 rows, which is most of what a shallow regression tree
    costs.

    `labels_q` is already quantized (DEVIATION 135); `scale` puts the leaf
    values back in the label's units.
    """
    error_checking(n_rows, n_cols, n_trees)
    validity_check(params)
    var dataset = upload_dataset(
        ctx, x_col_major, labels_q, n_rows, n_cols, 1
    )
    var ws = make_level_workspace(
        ctx,
        Int(params.max_batch_size),
        n_rows,
        n_cols,
        1,
        Int(n_sampled_cols_for(params, n_cols)),
        DEVICE_TPB,
    )
    var forest = Forest(1)
    for tree_id in range(Int(n_trees)):
        var row_ids = row_sample_for(n_rows, bootstrap)
        forest.trees.append(
            train_regression_device_resident(
                ctx, dataset, ws, scale, row_ids, n_rows, n_cols, params,
                Int32(tree_id), seed,
            )
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
