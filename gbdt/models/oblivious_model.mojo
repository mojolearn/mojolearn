"""The trained tree, and the ensemble of them.

PORT OF `catboost/cuda/models/oblivious_model.h` and `additive_model.h` at
CatBoost `54a8143a`. Transliterated. Do not improve.

Their structure is remarkably small:

    struct TObliviousTreeStructure { TVector<TBinarySplit> Splits; };

An oblivious tree applies the SAME split at every node of a level, so the
whole structure is one split per level. Depth `d` is `d` splits and `1 << d`
leaves, which is `GetDepth()` and `LeavesCount()` verbatim.

## The leaf index, and why ours already agrees with theirs

A row's leaf is the split outcomes read as a binary number. Level 0 is the
LEAST significant bit:

    leaf = sum over levels of (bit_l << l)

That falls out of the numbering `run_tree_layout` adopted from
`split_properties_helper.cpp:861` for a different reason entirely: the left
child KEEPS its parent's slot `i` and the right child is appended at
`leavesCount + i`. After `d` levels that puts a row with bits
`(b_0, b_1, ...)` in slot `sum b_l << l`, which is their prediction-time
convention. Adopting their growth numbering handed us their model numbering
for free, and it would not have if we had kept `2i` and `2i+1`.
"""


# --- EBinSplitType (`cuda/data/feature.h:21-24`) -------------------------
#
# THEIR ORDER AND THEIR VALUES, so a number read out of one of their
# structures means the same thing here.

comptime BIN_SPLIT_TAKE_BIN = 0
comptime BIN_SPLIT_TAKE_GREATER = 1


def bin_split_type_name(t: Int) -> String:
    if t == BIN_SPLIT_TAKE_BIN:
        return String("TakeBin")
    if t == BIN_SPLIT_TAKE_GREATER:
        return String("TakeGreater")
    return String("<unknown split type>")


@fieldwise_init
struct TBinarySplit(Copyable, ImplicitlyCopyable, Movable):
    """Their `TBinarySplit` (`cuda/data/feature.h:35-38`), all three
    members.

    `split_type` is their `EBinSplitType`: `TakeGreater` compares the bin
    against a border (`featureVal > value`) and `TakeBin` tests equality
    (`featureVal == value`), which is the one-hot predicate. Their
    `ToSplit` sets it from `manager.IsCat(props.FeatureId)`
    (`cuda/methods/helpers.cpp:164-170`) and every consumer switches on it
    (`add_oblivious_tree_model_doc_parallel.cpp:43`, `:139`).

    **It used to be absent, and the predicate came off the LAYOUT.** That
    is what their training-side apply does too, but theirs asserts the two
    agree (`CB_ENSURE(dataSet.IsOneHot(split.FeatureId))`) and ours could
    not, because the model did not know. A model read back from a file has
    no layout in hand until one is rebuilt from its own fold counts, so a
    predicate that lives only in the layout is a predicate the file does
    not carry.
    """

    var feature_id: Int32
    var bin_idx: Int32
    var split_type: Int32


struct TObliviousTreeStructure(Copyable, Movable):
    """Their `TObliviousTreeStructure` (`oblivious_model.h:9`)."""

    var splits: List[TBinarySplit]

    def __init__(out self):
        self.splits = List[TBinarySplit]()

    def get_depth(self) -> Int:
        """Their `GetDepth()`."""
        return len(self.splits)

    def leaves_count(self) -> Int:
        """Their `LeavesCount()`, `1 << GetDepth()`."""
        return 1 << self.get_depth()

    def has_split(self, candidate: TBinarySplit) -> Bool:
        """Their `HasSplit` (`oblivious_model.h:24-31`), which compares whole
        `TBinarySplit`s -- and their `operator==` ties all THREE members
        (`feature.h:59-61`), split type included."""
        for i in range(len(self.splits)):
            if (
                self.splits[i].feature_id == candidate.feature_id
                and self.splits[i].bin_idx == candidate.bin_idx
                and self.splits[i].split_type == candidate.split_type
            ):
                return True
        return False


struct TObliviousTreeModel(Copyable, Movable):
    """Their `TObliviousTreeModel` (`oblivious_model.h:57`).

    `LeafWeights` is carried because their constructor does and because a
    leaf's weight is what tells a reader whether a value is trustworthy or
    the artifact of three rows. `Dim` is 1 until multiclass lands.

    **`leaf_weights` is NEVER FILLED and that is a gap, not a choice.** Theirs
    is populated on both paths: the structure search writes
    `(*resultsLeafWeights)[leafId] = w` as it terminates
    (`greedy_search_helper.cpp:642`), `BuildTreeLikeModel` permutes it into
    leaf order beside the values (`model_builder.cpp:70`), and the estimator
    overwrites it again from `WriteWeights` (`doc_parallel_leaves_estimator.cpp:20,
    :40`). `run_tree_layout` returns leaf SIZES, which is a row count and not
    a weight, so filling the field from what `fit` has would be wrong whenever
    the rows carry weights. Closing it means returning the weight plane of
    `part_stats` alongside the values.
    """

    var structure: TObliviousTreeStructure
    var leaf_values: List[Float32]
    var leaf_weights: List[Float32]
    var dim: Int

    def __init__(out self, var structure: TObliviousTreeStructure):
        self.structure = structure^
        self.leaf_values = List[Float32]()
        self.leaf_weights = List[Float32]()
        self.dim = 1

    def get_structure(self) -> TObliviousTreeStructure:
        """Their `GetStructure()`."""
        return self.structure.copy()


struct TAdditiveModel(Copyable, Movable):
    """Their `TAdditiveModel<TWeakModel>` (`additive_model.h`), the ensemble.

    Prediction is the sum over weak models. The learning rate is already
    folded into the stored leaf values by their `Rescale(step)`
    (`doc_parallel_boosting.h:389-391`), so nothing here reapplies it.

    Their `SetBias` is not carried (`doc_parallel_boosting.h:527`). It exists
    to record `cursors->StartingPoint`, which is set only under
    `boost_from_average` or `RMSEWithUncertainty` (`:146-148`); both are
    refused by `CatBoostOptions.check()`, so the bias would be zero on every
    model this port can build. It has to appear the day either lands, because
    a model whose cursor started somewhere other than zero and does not say so
    predicts the residual rather than the target.
    """

    var weak_models: List[TObliviousTreeModel]

    def __init__(out self):
        self.weak_models = List[TObliviousTreeModel]()

    def add_weak_model(mut self, var model: TObliviousTreeModel):
        """Their `AddWeakModel`."""
        self.weak_models.append(model^)

    def size(self) -> Int:
        return len(self.weak_models)
